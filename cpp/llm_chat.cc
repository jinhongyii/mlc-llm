/*!
 *  Copyright (c) 2023 by Contributors
 * \file llm_chat.cc
 * \brief Implementation of llm chat.
 */
#define PICOJSON_USE_INT64
#define __STDC_FORMAT_MACROS

#include "llm_chat.h"

#include <picojson.h>
#include <tokenizers_cpp.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/memory_manager.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/ir/attrs.h>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <list>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>

#include "conversation.h"

namespace mlc {
namespace llm {

using tvm::Device;
using namespace tvm::runtime;

//----------------------------
// Tokenizers
//----------------------------
using tokenizers::Tokenizer;

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  ICHECK(!fs.fail()) << "Cannot open " << path;
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

std::unique_ptr<Tokenizer> TokenizerFromPath(const std::string& _path) {
  std::filesystem::path path(_path);
  std::filesystem::path sentencepiece;
  std::filesystem::path huggingface;
  CHECK(std::filesystem::exists(path)) << "Cannot find tokenizer via path: " << _path;
  if (std::filesystem::is_directory(path)) {
    sentencepiece = path / "tokenizer.model";
    huggingface = path / "tokenizer.json";
    // Check ByteLevelBPE
    {
      std::filesystem::path merges_path = path / "merges.txt";
      std::filesystem::path vocab_path = path / "vocab.json";
      std::filesystem::path added_tokens_path = path / "added_tokens.json";
      if (std::filesystem::exists(merges_path) && std::filesystem::exists(vocab_path) &&
          std::filesystem::exists(added_tokens_path)) {
        std::string vocab = LoadBytesFromFile(vocab_path.string());
        std::string merges = LoadBytesFromFile(merges_path.string());
        std::string added_tokens = LoadBytesFromFile(added_tokens_path.string());
        return Tokenizer::FromBlobByteLevelBPE(vocab, merges, added_tokens);
      }
    }
  } else {
    sentencepiece = path.parent_path() / "tokenizer.model";
    huggingface = path.parent_path() / "tokenizer.json";
  }
  if (std::filesystem::exists(sentencepiece)) {
    return Tokenizer::FromBlobSentencePiece(LoadBytesFromFile(sentencepiece.string()));
  }
  if (std::filesystem::exists(huggingface)) {
    return Tokenizer::FromBlobJSON(LoadBytesFromFile(huggingface.string()));
  }
  LOG(FATAL) << "Cannot find any tokenizer under: " << _path;
}

//------------------------------
// support functions
//------------------------------
inline size_t FindEffectiveUTF8Pos(const std::string& s) {
  int pos = s.size() - 1;
  for (; pos >= 0; pos--) {
    if ((s[pos] & 0x80) == 0x00) {
      return pos + 1;
    } else if (pos - 1 >= 0 && (s[pos - 1] & 0xE0) == 0xC0 && (s[pos] & 0xC0) == 0x80) {
      return pos + 1;
    } else if (pos - 2 >= 0 && (s[pos - 2] & 0xF0) == 0xE0 && (s[pos - 1] & 0xC0) == 0x80 &&
               (s[pos] & 0xC0) == 0x80) {
      return pos + 1;
    } else if (pos - 3 >= 0 && (s[pos - 3] & 0xF8) == 0xF0 && (s[pos - 2] & 0xC0) == 0x80 &&
               (s[pos - 1] & 0xC0) == 0x80 && (s[pos] & 0xC0) == 0x80) {
      return pos + 1;
    }
  }
  return pos + 1;
}

inline std::string Concat(const std::vector<std::string>& inputs) {
  std::ostringstream os;
  for (const auto& x : inputs) {
    os << x;
  }
  return os.str();
}

//------------------------------
// Chat module
//------------------------------
class LLMChatModule;

/*!
 * \brief Implements the chat conversation wrapper
 */
class LLMChat {
  friend class LLMChatModule;

 public:
  explicit LLMChat(DLDevice device) : device_(device) {}

  /*!
   * \return Text describing runtime stats.
   */
  std::string RuntimeStatsText() {
    std::ostringstream os;
    os << "prefill: " << std::setprecision(1) << std::fixed
       << this->prefill_total_tokens / (this->prefill_total_time + this->embed_total_time)
       << " tok/s"
       << ", decode: " << std::setprecision(1) << std::fixed
       << this->decode_total_tokens / this->decode_total_time << " tok/s";
    return os.str();
  }

  /*!
   * \brief Load JSON config and override options.
   * \param config_json A json config in picojson type that is partially specifies
   *        some of the options.
   * \param partial_update Whether it's a partial update or full update, if set to true,
   *        we perform a partial update on some of the provided options; if set to false, all
   *        options must be provided.
   * \note This function overrides existing configurations.
   */
  void LoadJSONOverride(const picojson::value& config_json, bool partial_update = false) {
    picojson::object config = config_json.get<picojson::object>();
    if (config.count("temperature")) {
      CHECK(config["temperature"].is<double>());
      this->temperature_ = config["temperature"].get<double>();
    } else {
      CHECK(partial_update) << "Key \"temperature\" not found.";
    }
    if (config.count("vocab_size")) {
      CHECK(config["vocab_size"].is<int64_t>());
      this->vocab_size_ = config["vocab_size"].get<int64_t>();
    } else {
      CHECK(partial_update) << "Key \"vocab_size\" not found.";
    }
    if(config.count("max_window_size")) {
      CHECK(config["max_window_size"].is<int64_t>());
      this->max_window_size_ = config["max_window_size"].get<int64_t>();
    } else {
      CHECK(partial_update) << "Key \"max_window_size\" not found.";
    }
    if(config.count("model_name")){
      CHECK(config["model_name"].is<std::string>());
      this->model_name_ = config["model_name"].get<std::string>();
    } else {
      CHECK(partial_update) << "Key \"model_name\" not found.";
    }
    if (config.count("repetition_penalty")) {
      CHECK(config["repetition_penalty"].is<double>());
      CHECK(this->repetition_penalty_ > 0) << "Repetition penalty must be a positive number!";
      this->repetition_penalty_ = config["repetition_penalty"].get<double>();
    } else {
      CHECK(partial_update) << "Key \"repetition_penalty\" not found.";
    }
    if (config.count("top_p")) {
      CHECK(config["top_p"].is<double>());
      this->top_p_ = config["top_p"].get<double>();
    } else {
      CHECK(partial_update) << "Key \"top_p\" not found.";
    }
    if (config.count("mean_gen_len")) {
      CHECK(config["mean_gen_len"].is<int64_t>());
      this->mean_gen_len_ = config["mean_gen_len"].get<int64_t>();
    } else {
      CHECK(partial_update) << "Key \"mean_gen_len\" not found.";
    }
    // NOTE: for backward compact
    // max gen len is optional
    if (config.count("max_gen_len")) {
      CHECK(config["max_gen_len"].is<int64_t>());
      this->max_gen_len_ = config["max_gen_len"].get<int64_t>();
    }
    if (config.count("shift_fill_factor")) {
      CHECK(config["shift_fill_factor"].is<double>());
      this->shift_fill_factor_ = config["shift_fill_factor"].get<double>();
    } else {
      CHECK(partial_update) << "Key \"shift_fill_factor\" not found.";
    }
    if (config.count("conv_template")) {
      ICHECK(config["conv_template"].is<std::string>());
      std::string conv_template = config["conv_template"].get<std::string>();
      this->conversation_ = Conversation::FromTemplate(conv_template);
      if (config.count("conv_config")) {
        // conv_config can override conv_template
        this->conversation_.LoadJSONOverride(config["conv_config"], true);
      }
    } else if (config.count("conv_config")) {
      // without conv template, conv_config needs to be a complete config
      this->conversation_.LoadJSONOverride(config["conv_config"], false);
    } else {
      CHECK(partial_update) << "Key \"conv_template\" and \"conv_config\" not found.";
    }
  }

  /*!
   * \brief Load JSON config and override options.
   * \param config_str A json config string that partially specifies some of the options.
   * \param partial_update Whether it's a partial update or full update, if set to true,
   *        we perform a partial update on some of the provided options; if set to false, all
   *        options must be provided.
   * \note This function overrides existing configurations.
   */
  void LoadJSONOverride(const std::string& config_str, bool partial_update = false) {
    picojson::value config_json;
    std::string err = picojson::parse(config_json, config_str);
    if (!err.empty()) {
      LOG(FATAL) << err;
      return;
    }
    LoadJSONOverride(config_json, partial_update);
  }

  std::string GetConfigJSON() const { return SerializeConfigToJSONValue().serialize(true); }

  /*!
   * \brief Reload model, tokenizers and configurations from the specified model path.
   * \param executable The module to reload.
   * \param model_path The path to search for models.
   * \param app_config_json The JSON string used to partially override the configuration loaded from
   * disk, default to empty string.
   */
  void Reload(String lib_path, String model_path, String app_config_json = "") {
    // Step 1. Set tokenizer.
    this->tokenizer_ = TokenizerFromPath(model_path);

    // Step 2. Initialize vm, we use the packed function mechanism
    // so there is no explicit abi dependency on these extra
    // classes other than basic tvm runtime.
    tvm::runtime::Module executable = tvm::runtime::Module::LoadFromFile(lib_path);
    auto f_func_exists_ = executable->GetFunction("func_exists");
    ICHECK(f_func_exists_.defined()) << "TVM runtime cannot find func_exists";

    auto fthreaded_session = tvm::runtime::Registry::Get("runtime.disco.SessionThreaded");
    ICHECK(fthreaded_session) << "TVM runtime cannot find runtime.disco.SessionThreaded";
    session_ = (*fthreaded_session)(1);

    auto fnccl_init = session_->GetGlobalFunc("runtime.disco.nccl.init_ccl");
    session_->CallPacked(fnccl_init, 0);

    auto fvm_load_module = session_->GetGlobalFunc("runtime.disco.load_vm_module");
    auto mod = session_->CallPacked(fvm_load_module, lib_path, device_);

    auto fmodule_get_function = session_->GetGlobalFunc("runtime.ModuleGetFunction");

    prefill_func_ = session_->CallPacked(fmodule_get_function, mod, "prefill", false);
    func_exists_[prefill_func_] = f_func_exists_("prefill");
    embed_func_ = session_->CallPacked(fmodule_get_function, mod, "embed", false);
    func_exists_[embed_func_] = f_func_exists_("embed");
    prefill_with_embed_func_ = session_->CallPacked(fmodule_get_function, mod, "prefill_with_embed",
                                                    false);
    func_exists_[prefill_with_embed_func_] = f_func_exists_("prefill_with_embed");
    decode_func_ = session_->CallPacked(fmodule_get_function, mod, "decode", false);
    func_exists_[decode_func_] = f_func_exists_("decode");
    encoding_without_cache_func_ =  
        session_->CallPacked(fmodule_get_function, mod, "encoding_without_cache", false);
    func_exists_[encoding_without_cache_func_] = f_func_exists_("encoding_without_cache");
    softmax_func_ = session_->CallPacked(fmodule_get_function, mod, "softmax_with_temperature", false);
    func_exists_[softmax_func_] = f_func_exists_("softmax_with_temperature");
    tuple_getitem_func_ = session_->GetGlobalFunc("vm.builtin.tuple_getitem");
    auto fsample_topp_from_prob_ptr =
        tvm::runtime::Registry::Get("vm.builtin.sample_top_p_from_prob");
    ICHECK(fsample_topp_from_prob_ptr)
        << "Cannot find env function vm.builtin.sample_top_p_from_prob";
    fsample_topp_from_prob_ = *fsample_topp_from_prob_ptr;
    auto fsample_topp_from_logits_ptr =
        tvm::runtime::Registry::Get("vm.builtin.sample_top_p_from_logits");
    ICHECK(fsample_topp_from_logits_ptr)
        << "Cannot find env function vm.builtin.sample_top_p_from_logits";
    fsample_topp_from_logits_ = *fsample_topp_from_logits_ptr;

    // Step 3. Load params in nd-array cache.

    auto fcreate_shard_loader = session_->GetGlobalFunc("runtime.disco.ShardLoader");
    auto fload_shard = session_->GetGlobalFunc("runtime.disco.ShardLoaderLoadAll");
    // FIXME: fix the shard info
    picojson::object shard_info;
    std::string shard_info_str = picojson::value(shard_info).serialize(true);

    std::string json_path = model_path + "/ndarray-cache.json";
    auto ndarray_cache_metadata = LoadBytesFromFile(json_path);
    PackedFunc tmp(nullptr);
    auto loader = session_->CallPacked(fcreate_shard_loader,json_path,
                                       ndarray_cache_metadata, shard_info_str, tmp);
    params_ = session_->CallPacked(fload_shard, loader);
    // Step 4. KV cache creation.

    auto fcreate_kv_cache = session_->CallPacked(fmodule_get_function, mod, "create_kv_cache",
                                                 false);
    kv_cache_ = session_->CallPacked(fcreate_kv_cache);

    fkvcache_array_popn_ = session_->GetGlobalFunc("vm.builtin.attention_kv_cache_array_popn");
    // Step 5. KV cache reset.
    reset_kv_cache_func_ = session_->CallPacked(fmodule_get_function, mod, "reset_kv_cache",
                                                false);
    if (!f_func_exists_("reset_kv_cache")) {
      auto attention_kv_cache_array_clear_ptr = session_->GetGlobalFunc("vm.builtin.attention_kv_cache_array_clear");

      reset_kv_cache_func_ = attention_kv_cache_array_clear_ptr;
      support_backtracking_kv_ = true;
    } else {
      // if there is a customized reset kv
      // then it may not be the typical transformer model
      // and we disable backtracking kv feature
      support_backtracking_kv_ = false;
    }

    // Step 6. Process config json string.
    std::ifstream config_istream((model_path + "/mlc-chat-config.json").c_str());
    std::ostringstream config_ostream;
    ICHECK(config_istream);
    config_ostream << config_istream.rdbuf();
    std::string config_str = config_ostream.str();
    LoadJSONOverride(config_str, false);

    // Step 7. Process metadata

    // fixme: load from json
    this->model_name_ = "Llama-2-7b-chat-hf-q4f16_1";
    this->max_window_size_ = 2048;

    // Step 8. Override configuration from app_config_json.
    if (!app_config_json.empty()) {
      LoadJSONOverride(app_config_json, true);
    }

    this->ResetChat();
  }

  // // TODO: remove the legacy initialization func after updating app and web sides.
  // void InitChatLegacy(String conv_template, double temperature, double top_p, int64_t mean_gen_len,
  //                     double shift_fill_factor) {
  //   // Process metadata
  //   std::string metadata_str = this->GetMetadata();
  //   picojson::value metadata_info;
  //   picojson::parse(metadata_info, metadata_str);
  //   auto metadata = metadata_info.get<picojson::object>();
  //   ICHECK(metadata["model_name"].is<std::string>());
  //   ICHECK(metadata["max_window_size"].is<int64_t>());
  //   ICHECK(metadata["add_prefix_space"].is<bool>());
  //   this->model_name_ = metadata["model_name"].get<std::string>();
  //   this->max_window_size_ = metadata["max_window_size"].get<int64_t>();
  //   if (this->max_window_size_ == -1) {
  //     this->max_window_size_ = std::numeric_limits<int64_t>::max();
  //   }
  //   this->conversation_ = Conversation::FromTemplate(conv_template);
  //   this->temperature_ = temperature;
  //   this->top_p_ = top_p;
  //   this->mean_gen_len_ = mean_gen_len;
  //   this->shift_fill_factor_ = shift_fill_factor;
  //   this->ResetChat();
  // }

  void ResetChat() {
    // TODO(mlc-team): add conversation_.Reset to preserve system prompt
    // and initial message.
    // this->conversation_ = Conversation::Create(this->conversation_.conv_template);
    this->conversation_.Reset();
    this->ResetRuntimeStats();
    this->ResetKVCache();
    this->total_seq_len_ = 0;
  }

  /*! \brief reset the runtime stats. */
  void ResetRuntimeStats() {
    this->prefill_total_tokens = 0;
    this->decode_total_tokens = 0;
    this->embed_total_time = 0;
    this->prefill_total_time = 0;
    this->decode_total_time = 0;
    this->sample_total_time = 0;
  }

  static std::string GetConcatPrompt(const std::vector<std::string>& prompt_array,
                                     size_t prefix_end, size_t suffix_start) {
    std::ostringstream os;
    for (size_t i = 0; i < prefix_end; ++i) {
      os << prompt_array[i];
    }
    for (size_t i = suffix_start; i < prompt_array.size(); ++i) {
      os << prompt_array[i];
    }
    return os.str();
  }

  /**
   * \brief Get input tokens based on history
   * \param place_in_prompt The place of the input message in the prompt.
   */
  std::vector<int32_t> GetInputTokens(PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    std::vector<int32_t> tokens;
    std::vector<std::string> prompts;

    if (this->total_seq_len_ == 0) {
      prompts = this->conversation_.GetPromptArray(place_in_prompt);
      if (this->conversation_.add_bos) {
        tokens.insert(tokens.begin(), bos_token_id_);
      }
      if (this->conversation_.prefix_tokens.size() != 0) {
        tokens.insert(tokens.begin(), this->conversation_.prefix_tokens.begin(),
                      this->conversation_.prefix_tokens.end());
      }
    } else {
      prompts = this->conversation_.GetPromptArrayLastRound(place_in_prompt);
    }
    // first try to encode all
    std::string all_prompt = GetConcatPrompt(prompts, 0, 0);
    std::vector<int32_t> encoded = this->tokenizer_->Encode(all_prompt);
    tokens.insert(tokens.end(), encoded.begin(), encoded.end());
    if (this->total_seq_len_ + tokens.size() + this->mean_gen_len_ < this->max_window_size_) {
      return tokens;
    }
    // need shift window and re-encode
    this->total_seq_len_ = 0;
    this->ResetKVCache();
    tokens.clear();
    if (this->conversation_.add_bos) {
      tokens.insert(tokens.begin(), bos_token_id_);
    }
    if (this->conversation_.prefix_tokens.size() != 0) {
      tokens.insert(tokens.begin(), this->conversation_.prefix_tokens.begin(),
                    this->conversation_.prefix_tokens.end());
    }
    std::vector<std::string> all_prompts = this->conversation_.GetPromptArray();
    // get estimate of the fragment
    size_t ctx_length = this->tokenizer_->Encode(all_prompts[0]).size();
    size_t start_re_encode_pos = 0;
    for (int i = all_prompts.size() - 1; i > 0; i -= 2) {
      ctx_length += this->tokenizer_->Encode(all_prompts[i]).size();
      if (ctx_length >= this->shift_fill_factor_ * this->max_window_size_ &&
          i + 2 < all_prompts.size()) {
        start_re_encode_pos = i;
        break;
      }
    }
    // keep system
    if (this->conversation_.system.empty()) {
      all_prompt = GetConcatPrompt(prompts, 0, start_re_encode_pos);
    } else {
      all_prompt = GetConcatPrompt(prompts, 1, start_re_encode_pos);
    }
    encoded = this->tokenizer_->Encode(all_prompt);
    tokens.insert(tokens.end(), encoded.begin(), encoded.end());
    if (tokens.size() >= this->max_window_size_) {
      LOG(WARNING)
          << "The prompt tokens are more than `max_window_size`, the input will be truncated.";
      ICHECK_GT(this->max_window_size_, this->mean_gen_len_);
      std::vector<int32_t> truncated_tokens(
          tokens.end() - (this->max_window_size_ - this->mean_gen_len_), tokens.end());
      return truncated_tokens;
    } else if (tokens.size() + this->mean_gen_len_ >= this->max_window_size_) {
      LOG(WARNING)
          << "The prompt tokens are too long and the generated text may be incomplete, due to "
             "limited `max_window_size`. ";
    }
    return tokens;
  }

  // get statically allocated input token
  NDArray GetInputTokenNDArray(const std::vector<int32_t>& token_ids) {
    // try realloc
    if (!input_token_ids_.defined()) {
      int64_t init_size = 2048;
      while (init_size < static_cast<int64_t>(token_ids.size())) {
        init_size *= 2;
      }
      input_token_ids_ = NDArray::Empty({1, init_size}, DataType::Int(32), device_);
    } else {
      int64_t init_size = input_token_ids_->shape[1];
      while (init_size < static_cast<int64_t>(token_ids.size())) {
        init_size *= 2;
      }
      if (init_size != input_token_ids_->shape[1]) {
        input_token_ids_ = NDArray::Empty({1, init_size}, DataType::Int(32), device_);
      }
    }
    ICHECK_LE(token_ids.size(), input_token_ids_->shape[1]) << "Input tokens exceed window size";
    NDArray view = input_token_ids_.CreateView(
        ShapeTuple({1, static_cast<int64_t>(token_ids.size())}), input_token_ids_->dtype);
    if (token_ids.size() > 0) {
      view.CopyFromBytes(token_ids.data(), token_ids.size() * sizeof(int32_t));
    }
    return view;
  }


  std::vector<int32_t> PrepareBeforeEmbedding(std::string inp, bool append_conversation = true,
                                              PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    if (conversation_.separator_style == SeparatorStyle::kLM ||
        conversation_.separator_style == SeparatorStyle::kCodeCompletion) {
      this->ResetChat();
    }
    if (reset_stats_per_prefill_) {
      this->ResetRuntimeStats();
    }
    output_ids_.clear();
    appeared_token_ids_.clear();
    output_message_.clear();
    stop_triggered_ = false;
    if (append_conversation) {
      conversation_.AppendMessage(conversation_.roles[0], inp);
      conversation_.AppendReplyHeader(conversation_.roles[1]);
    }

    return this->GetInputTokens(place_in_prompt);
  }

  DRef CopyToWorker0(const NDArray& host_array) { 
    Device dev {DLDeviceType(0), 0};
    auto func = session_->GetGlobalFunc("runtime.disco.empty");
    ShapeTuple shape = host_array.Shape();
    DataType dtype = host_array.DataType();
    DRef dref = session_->CallPacked(func, shape, dtype ,dev);
    session_->CopyToWorker0(host_array, dref);
    return dref;
  }

  /*!
   * \brief Given the text input, generate the embedding of the tokenized prompt.
   * \param inp The input text string.
   * \param append_conversation Whether to append the input message to conversation.
   * \param place_in_prompt The place of the input message in the prompt.
   * \return the embedding of the tokenized prompt.
   */
  DRef EmbedStep(std::string inp, bool append_conversation = true,
                    PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    std::vector<int32_t> prompt_tokens =
        PrepareBeforeEmbedding(inp, append_conversation, place_in_prompt);
    int64_t token_len = static_cast<int64_t>(prompt_tokens.size());
    if (token_len == 0) {
      auto empty_func = session_->GetGlobalFunc("runtime.disco.empty");
      return session_->CallPacked(empty_func, ShapeTuple({}), DataType::Float(32), device_);
    }

    CHECK(func_exists_[embed_func_]) << "In order to use the embedding functionality, make sure you "
                                    "build the model in MLC-LLM with `sep_embed` option on.";
    auto tstart = std::chrono::high_resolution_clock::now();

    NDArray input_data = this->GetInputTokenNDArray(prompt_tokens);
    DRef input_dref = CopyToWorker0(input_data);
    DRef embedding = session_->CallPacked(embed_func_, input_dref, params_);

    int32_t new_seq_len = total_seq_len_ + token_len;
    total_seq_len_ = new_seq_len;

    auto tend = std::chrono::high_resolution_clock::now();

    this->embed_total_time += static_cast<double>((tend - tstart).count()) / 1e9;

    return embedding;
  }

  /*!
   * \brief Prefill given embeddings. Can optionally decode the output next token.
   * \param embedding The embedding to prefill with.
   * \param decode_next_token Whether to decode next token.
   */
  void PrefillWithEmbedStep(DRef embedding, bool decode_next_token = true) {
    LOG(FATAL) << "disco is not compatible with embed step";
    NDArray embedding_ndarray;
    session_->CopyFromWorker0(embedding_ndarray, embedding);
    session_->SyncWorker(0);
    if (embedding_ndarray.Shape().size() == 0) {
      return;
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    int64_t token_len = embedding_ndarray.Shape()[1];
    NDArray logits_on_device = this->ForwardEmbeddings(embedding, total_seq_len_);

    if (!decode_next_token) {
      auto tend = std::chrono::high_resolution_clock::now();
      this->prefill_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
      this->prefill_total_tokens += token_len;
      return;
    }

    int32_t next_token = this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_);

    auto tend = std::chrono::high_resolution_clock::now();

    this->prefill_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    this->prefill_total_tokens += token_len;
    this->ProcessNextToken(next_token);
  }

  /*!
   * \brief Generate the next token given a prompt. Can optionally decode the output next token.
   * \param inp The input text string.
   * \param append_conversation Whether to append the input message to conversation.
   * \param decode_next_token Whether to decode next token.
   * \param place_in_prompt The place of the input message in the prompt.
   */
  void PrefillStep(std::string inp, bool append_conversation = true, bool decode_next_token = true,
                   PlaceInPrompt place_in_prompt = PlaceInPrompt::kAll) {
    if (func_exists_[embed_func_] && func_exists_[prefill_with_embed_func_]) {
      // Temporarily placed inside `PrefillStep` for compatibility in transition.
      // Will be separated out in the future.
      DRef embedding = EmbedStep(inp, append_conversation, place_in_prompt);
      PrefillWithEmbedStep(embedding, decode_next_token);
      return;
    }
    std::vector<int32_t> prompt_tokens =
        this->PrepareBeforeEmbedding(inp, append_conversation, place_in_prompt);
    int64_t token_len = static_cast<int64_t>(prompt_tokens.size());
    if (token_len == 0) return;

    auto tstart = std::chrono::high_resolution_clock::now();

    int32_t new_seq_len = total_seq_len_ + token_len;
    NDArray logits_on_device = this->ForwardTokens(prompt_tokens, new_seq_len);
    // this->UpdateLogitsOrProbOnCPUSync(logits_on_device);
    // TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    // // print first few logits for eyeballs
    // std::ostringstream os;
    // for (int i = 0; i < 10; ++i) {
    //   if (i != 0) os << ", ";
    //   os << static_cast<float*>(logits_on_cpu_->data)[i];
    // }
    // LOG(INFO) << "logits[:10] =[" << os.str() << "]";
    total_seq_len_ = new_seq_len;

    if (!decode_next_token) {
      auto tend = std::chrono::high_resolution_clock::now();
      this->prefill_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
      this->prefill_total_tokens += token_len;
      return;
    }

    int32_t next_token = this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_);

    auto tend = std::chrono::high_resolution_clock::now();

    this->prefill_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    this->prefill_total_tokens += token_len;
    this->ProcessNextToken(next_token);
  }

  void DecodeStep() {
    ICHECK(!output_ids_.empty());
    int32_t last_token = output_ids_.back();
    tvm::runtime::NDArray input_data = GetInputTokenNDArray({last_token});

    auto tstart = std::chrono::high_resolution_clock::now();

    NDArray logits_on_device = this->ForwardTokens({last_token}, total_seq_len_ + 1);
    total_seq_len_ += 1;

    int32_t next_token = this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_);

    auto tend = std::chrono::high_resolution_clock::now();

    this->decode_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    this->decode_total_tokens += 1;
    this->ProcessNextToken(next_token);
  }

  bool Stopped() { return stop_triggered_; }

  std::string GetMessage() {
    // remove non-utf8 characters
    size_t effective_end = FindEffectiveUTF8Pos(output_message_);
    while (effective_end > 0 && output_message_[effective_end - 1] == '\n') {
      --effective_end;
    }
    size_t effective_begin = 0;
    while (effective_begin < effective_end && output_message_[effective_begin] == ' ') {
      ++effective_begin;
    }
    std::string cropped_message =
        output_message_.substr(effective_begin, effective_end - effective_begin);
    return cropped_message;
  }

  // do some quick evaluation of the pipeline
  void Evaluate(int64_t token_len, int64_t generate_len) {
    this->ResetKVCache();
    std::vector<int32_t> tokens;
    for (int i = 0; i < token_len - 1; ++i) {
      tokens.push_back(2);
    }
    tokens.insert(tokens.begin(), bos_token_id_);

    std::vector<int32_t> first_sample_data = {6234};

    // warm up: skip first run
    this->ForwardTokens(tokens, token_len);
    this->ForwardTokens(first_sample_data, token_len + 1);
    this->ResetKVCache();

    // encoding
    auto encoding_start = std::chrono::high_resolution_clock::now();
    this->ForwardTokens(tokens, token_len);
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
    auto encoding_end = std::chrono::high_resolution_clock::now();
    double encoding_ms = static_cast<double>((encoding_end - encoding_start).count()) / 1e6;
    LOG(INFO) << "encoding-time=" << encoding_ms << "ms, ";

    double decoding_ms_total = 0;
    // start encoding
    for (int i = 0; i < generate_len; ++i) {
      auto decoding_start = std::chrono::high_resolution_clock::now();
      this->UpdateLogitsOrProbOnCPUSync(this->ForwardTokens(first_sample_data, token_len + i + 1));
      TVMSynchronize(device_.device_type, device_.device_id, nullptr);
      auto decoding_end = std::chrono::high_resolution_clock::now();
      double decoding_ms = static_cast<double>((decoding_end - decoding_start).count()) / 1e6;
      decoding_ms_total += decoding_ms;
      LOG(INFO) << "[i: " << token_len + i + 1 << "] decoding-time=" << decoding_ms << "ms"
                << " tok/s: " << 1000.0 * (i + 1) / decoding_ms_total << ".";
    }
  }

  std::string RawGenerate(std::string prompt, int64_t generate_len) {
    CHECK_GE(generate_len, 0) << "The input generate is expected to be non-negative.";

    this->ResetKVCache();
    this->ResetRuntimeStats();

    std::vector<int32_t> tokens = tokenizer_->Encode(prompt);
    int64_t input_length = tokens.size();

    NDArray logits_on_device;
    // prefill
    {
      auto tstart = std::chrono::high_resolution_clock::now();
      logits_on_device = this->ForwardTokens(tokens, tokens.size());
      tokens.push_back(this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_));
      auto tend = std::chrono::high_resolution_clock::now();

      this->prefill_total_time = static_cast<double>((tend - tstart).count()) / 1e9;
      this->prefill_total_tokens = input_length;
    }

    // decode
    {
      auto tstart = std::chrono::high_resolution_clock::now();
      for (int64_t len = 1; len < generate_len; ++len) {
        logits_on_device = this->ForwardTokens({tokens.back()}, tokens.size());
        tokens.push_back(this->SampleTokenFromLogits(logits_on_device, temperature_, top_p_));
      }
      auto tend = std::chrono::high_resolution_clock::now();

      this->decode_total_time = static_cast<double>((tend - tstart).count()) / 1e9;
      this->decode_total_tokens = generate_len;
    }

    std::string output = tokenizer_->Decode({tokens.begin() + input_length, tokens.end()});
    return output;
  }

 private:
  picojson::value SerializeConfigToJSONValue() const {
    picojson::object config;
    config["temperature"] = picojson::value(this->temperature_);
    config["repetition_penalty"] = picojson::value(this->repetition_penalty_);
    config["top_p"] = picojson::value(this->top_p_);
    config["mean_gen_len"] = picojson::value(this->mean_gen_len_);
    config["max_gen_len"] = picojson::value(this->max_gen_len_);
    config["shift_fill_factor"] = picojson::value(this->shift_fill_factor_);
    config["conv_config"] = this->conversation_.SerializeToJSON();
    return picojson::value(config);
  }
  /*!
   * \brief Sample output token from logits on device
   */
  int32_t SampleTokenFromLogits(NDArray logits_on_device, float temperature, float top_p) {
    if (repetition_penalty_ == 1.0f) {
      if (temperature_ < 1e-6f) {
        this->UpdateLogitsOrProbOnCPUSync(logits_on_device);
      } else {
        this->UpdateLogitsOrProbOnCPUSync(this->Softmax(logits_on_device, temperature_));
      }
    } else {
      this->UpdateLogitsOrProbOnCPUSync(logits_on_device);
      this->ApplyRepetitionPenaltyOnCPU();
      if (temperature_ >= 1e-6f) {
        this->ApplySoftmaxWithTemperatureOnCPU();
      }
    }
    auto tstart = std::chrono::high_resolution_clock::now();
    int next_token;
    if (temperature_ < 1e-6f) {
      next_token = this->SampleFromLogitsOnCPU();
    } else {
      next_token = this->SampleFromProbOnCPU();
    }
    auto tend = std::chrono::high_resolution_clock::now();
    this->sample_total_time += static_cast<double>((tend - tstart).count()) / 1e9;
    return next_token;
  }

  /*!
   * \brief Add a generated token and check for stop condition.
   * \param next_token The next token.
   */
  void ProcessNextToken(int32_t next_token) {
    ICHECK(!stop_triggered_) << "Cannot call process when it is stopped";

    stop_triggered_ =
        std::any_of(this->conversation_.stop_tokens.begin(), this->conversation_.stop_tokens.end(),
                    [next_token](int32_t token) { return token == next_token; });

    if (!stop_triggered_) {
      output_ids_.push_back(next_token);
      appeared_token_ids_.insert(next_token);
    }

    output_message_ = tokenizer_->Decode(output_ids_);

    if (!conversation_.stop_str.empty()) {
      size_t stop_pos = output_message_.rfind(conversation_.stop_str);
      if (stop_pos != std::string::npos) {
        stop_triggered_ = true;
        if (support_backtracking_kv_) {
          // back tracking, find the first set of token that is smaller
          // than the length
          size_t backoff = 0;
          for (; backoff < output_ids_.size(); ++backoff) {
            output_ids_.pop_back();
            output_message_ = tokenizer_->Decode(output_ids_);
            if (output_message_.length() <= stop_pos) break;
          }
          // resize kv to remove the context
          session_->CallPacked(fkvcache_array_popn_, kv_cache_, backoff);
          total_seq_len_ -= backoff;
        }
      }
    }

    if (static_cast<int64_t>(output_ids_.size()) >= max_gen_len_) {
      stop_triggered_ = true;
    } else if (total_seq_len_ >= max_window_size_) {
      stop_triggered_ = true;
    }
    if (stop_triggered_) {
      conversation_.FinishReply(output_message_);
    }
  }

  // run forward compute
  NDArray ForwardTokens(std::vector<int32_t> input_tokens, int64_t cur_pos) {
    NDArray ret_ndarray = NDArray::Empty({1, 1, vocab_size_}, DataType::Float(32), device_);
    if (input_tokens.size() > 1 && func_exists_[prefill_func_]) {
      NDArray input_data = this->GetInputTokenNDArray(input_tokens);
      DRef input_dref = CopyToWorker0(input_data);
      tvm::runtime::ShapeTuple cur_pos_shape = {cur_pos};
      DRef ret = session_->CallPacked(prefill_func_, input_dref, cur_pos_shape, kv_cache_,
                                      params_);
      ret = session_->CallPacked(tuple_getitem_func_, ret, 0);
      session_->CopyFromWorker0(ret_ndarray, ret);
      session_->SyncWorker(0);
    } else {
      // running decode function when prefill is not available
      for (int i = 0; i < input_tokens.size(); ++i) {
        NDArray input_data = this->GetInputTokenNDArray({input_tokens[i]});
        int64_t pos = cur_pos + i + 1 - input_tokens.size();
        DRef input_dref = CopyToWorker0(input_data);
        tvm::runtime::ShapeTuple pos_shape = {pos};
        DRef ret = session_->CallPacked(decode_func_, input_dref, pos_shape, kv_cache_, params_);
        ret = session_->CallPacked(tuple_getitem_func_, ret, 0);
        session_->CopyFromWorker0(ret_ndarray, ret);
        session_->SyncWorker(0);
      }
    }
    return ret_ndarray;
  }

  // run forward compute with embeddings
  NDArray ForwardEmbeddings(DRef embeddings, int64_t cur_pos) {
    NDArray ret_ndarray = NDArray::Empty({1, 1, vocab_size_}, DataType::Float(32), device_);
    tvm::runtime::ShapeTuple cur_pos_shape = {cur_pos};
    DRef ret = session_->CallPacked(prefill_with_embed_func_, embeddings, cur_pos_shape, kv_cache_, params_);
    ret = session_->CallPacked(tuple_getitem_func_, ret, 0);
    session_->CopyFromWorker0(ret_ndarray, ret);
    session_->SyncWorker(0);
    return ret_ndarray;
  }

  NDArray Softmax(NDArray input, float temperature) {
    NDArray ret_ndarray = NDArray::Empty(input.Shape(), DataType::Float(32), device_);
    NDArray temperature_arr = NDArray::Empty({}, DataType::Float(32), device_);
    temperature_arr.CopyFromBytes(&temperature, sizeof(float));
    DRef input_dref = CopyToWorker0(input);
    DRef temperature_dref = CopyToWorker0(temperature_arr);
    DRef ret = session_->CallPacked(softmax_func_, input_dref, temperature_dref);
    session_->CopyFromWorker0(ret_ndarray, ret);
    session_->SyncWorker(0);
    return ret_ndarray;
  }

  void ApplyRepetitionPenaltyOnCPU() {
    CHECK(logits_on_cpu_.defined()) << "Logits on CPU not defined!";
    CHECK(logits_on_cpu_.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
    float* logits_raw_data = static_cast<float*>(logits_on_cpu_->data);
    for (const int32_t& token_id : this->appeared_token_ids_) {
      if (logits_raw_data[token_id] <= 0) {
        logits_raw_data[token_id] *= this->repetition_penalty_;
      } else {  // logits > 0
        logits_raw_data[token_id] /= this->repetition_penalty_;
      }
    }
  }

  void ApplySoftmaxWithTemperatureOnCPU() {
    CHECK(logits_on_cpu_.defined()) << "Logits on CPU not defined!";
    CHECK(logits_on_cpu_.DataType() == DataType::Float(32)) << "Logits data type is not float32!";
    int vocab_size = logits_on_cpu_->shape[logits_on_cpu_->ndim - 1];
    float* logits_raw_data = static_cast<float*>(logits_on_cpu_->data);
    float m = std::numeric_limits<float>::min();
    float inv_temp = 1.0f / this->temperature_;
    double d = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
      float x = logits_raw_data[i] * inv_temp;
      float m_prev = m;
      m = std::max(m, x);
      d = d * std::exp(m_prev - m) + std::exp(x - m);
    }
    for (int i = 0; i < vocab_size; ++i) {
      float x = logits_raw_data[i] * inv_temp;
      logits_raw_data[i] = std::exp(x - m) / d;
    }
  }

  void UpdateLogitsOrProbOnCPUSync(NDArray logits_or_prob) {
    if (!logits_on_cpu_.defined()) {
      logits_on_cpu_ = logits_or_prob.CopyTo(DLDevice{kDLCPU, 0});
    } else {
      ICHECK_EQ(logits_on_cpu_->shape[0], logits_or_prob->shape[0])
          << "Expect size of logits remain unchanged";
      logits_on_cpu_.CopyFrom(logits_or_prob);
    }
    TVMSynchronize(device_.device_type, device_.device_id, nullptr);
  }

  // Clear kv cache
  void ResetKVCache() { session_->CallPacked(reset_kv_cache_func_, kv_cache_); }

  void ProcessSystemPrompts() {
    this->PrefillStep(/*inp=*/"", /*append_conversation=*/false, /*decode_next_token=*/false);
  }

  // Utils
  static double GetRandomNumber() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen);
  }

  int32_t SampleFromLogitsOnCPU() {
    ICHECK(logits_on_cpu_.defined()) << "logits_on_cpu_ is not defined";
    ICHECK_EQ(logits_on_cpu_->ndim, 3) << "logits_on_cpu_ should be 3D";
    ICHECK_EQ(logits_on_cpu_->shape[0], 1) << "logits_on_cpu_ should be 1 batch";
    return fsample_topp_from_logits_(logits_on_cpu_, top_p_, temperature_, GetRandomNumber());
  }

  int32_t SampleFromProbOnCPU() {
    ICHECK(logits_on_cpu_.defined()) << "logits_on_cpu_ is not defined";
    ICHECK_EQ(logits_on_cpu_->ndim, 3) << "logits_on_cpu_ should be 3D";
    ICHECK_EQ(logits_on_cpu_->shape[0], 1) << "logits_on_cpu_ should be 1 batch";
    return fsample_topp_from_prob_(logits_on_cpu_, top_p_, GetRandomNumber());
  }

  //----------------------------
  // Statistics
  //----------------------------
  bool reset_stats_per_prefill_ = true;
  double embed_total_time = 0;
  double decode_total_time = 0;
  double sample_total_time = 0;
  double prefill_total_time = 0;
  int64_t decode_total_tokens = 0;
  int64_t prefill_total_tokens = 0;
  //----------------------------
  // Conversation
  //----------------------------
  // model name
  std::string model_name_;
  // conversation
  Conversation conversation_;
  // total sequence len,
  int64_t total_seq_len_{0};
  // max window size, mean generation length
  int64_t max_window_size_{768}, mean_gen_len_{128}, max_gen_len_{512};
  // vocab size
  int64_t vocab_size_;
  // shift window fill factor
  double shift_fill_factor_{0.3};
  // temperature
  double temperature_{0.8};
  // repetition penalty
  double repetition_penalty_{1.0};
  // top_p
  double top_p_{0.95};
  // output ids till now (refresh after encoding step)
  std::vector<int32_t> output_ids_;
  // appeared token ids till now (refresh after encoding step)
  std::unordered_set<int32_t> appeared_token_ids_;
  // output message till now (refresh after encoding step)
  std::string output_message_;
  // Whether encounter stop str
  bool stop_triggered_{false};
  // Whether we support rollback kv
  bool support_backtracking_kv_ = true;
  //----------------------------
  // Tokenizer
  //----------------------------
  // internal tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;
  // bos token
  int32_t bos_token_id_{1};
  // eos token id
  int32_t eos_token_id_{2};
  //----------------------------
  // TVM related states
  //----------------------------
  // runtime device
  Device device_;
  // encoding function
  DRef prefill_func_{nullptr};
  // embedding function
  DRef embed_func_{nullptr};
  // encoding using embedding function
  DRef prefill_with_embed_func_{nullptr};
  // decoding function
  DRef decode_func_{nullptr};
  // encoding without cache
  DRef encoding_without_cache_func_{nullptr};
  // softmax
  DRef softmax_func_{nullptr};
  // reset kv cache
  DRef reset_kv_cache_func_{nullptr};
  // tuple get item
  DRef tuple_getitem_func_{nullptr};
  // sample top p from logits
  PackedFunc fsample_topp_from_logits_;
  // sample top p from prob
  PackedFunc fsample_topp_from_prob_;
  // pop n entries from kvcache
  DRef fkvcache_array_popn_{nullptr};
  // input token id
  NDArray input_token_ids_{nullptr};
  // local params
  DRef params_{nullptr};
  // KV cache
  DRef kv_cache_{nullptr};
  // Temp logits on cpu
  NDArray logits_on_cpu_{nullptr};
  // disco session
  tvm::runtime::Session session_{nullptr};
  // map of which function exists
  std::unordered_map<DRef, int, ObjectPtrHash, ObjectPtrEqual> func_exists_;
};

/*!
 * \brief A chat module implementation that exposes
 *  the functions as tvm::runtime::Module.
 *
 * We do it so that the module is accessible to any
 * language that tvm runtime can access.
 */
class LLMChatModule : public ModuleNode {
 public:
  // clear global memory manager
  static void ClearGlobalMemoryManager() {
    // Step 0. Clear the previously allocated memory.
    const PackedFunc* fclear_memory_manager =
        tvm::runtime::Registry::Get("vm.builtin.memory_manager.clear");
    ICHECK(fclear_memory_manager) << "Cannot find env function vm.builtin.memory_manager.clear";
    (*fclear_memory_manager)();
  }

  // overrides
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "reload") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        chat_ = nullptr;
        ClearGlobalMemoryManager();
        chat_ = std::make_unique<LLMChat>(LLMChat(device_));
        ICHECK(2 <= args.size() && args.size() <= 3);
        if (args.size() == 2) {
          // args: executable, model_path
          chat_->Reload(args[0], args[1]);
        } else if (args.size() == 3) {
          // args: executable, model_path, app_config_json (used for overriding config)
          chat_->Reload(args[0], args[1], args[2]);
        }
      });
    } else if (name == "unload") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        chat_ = nullptr;
        ClearGlobalMemoryManager();
      });
    } else if (name == "evaluate") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 2);
        GetChat()->Evaluate(args[0], args[1]);
      });
    } else if (name == "raw_generate") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 2);
        std::string s = GetChat()->RawGenerate(args[0], args[1]);
        *rv = s;
      });
    } else if (name == "prefill") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK(1 <= args.size() && args.size() <= 3);
        if (args.size() == 1) {
          // args: inp (with decode_next_token = true, place_in_prompt = kAll)
          GetChat()->PrefillStep(args[0]);
        } else if (args.size() == 2) {
          // args: inp, decode_next_token (with place_in_prompt = kAll)
          GetChat()->PrefillStep(args[0], true, args[1]);
        } else if (args.size() == 3) {
          // args: inp, decode_next_token, place_in_prompt
          PlaceInPrompt place_in_prompt = static_cast<PlaceInPrompt>(static_cast<int>(args[2]));
          GetChat()->PrefillStep(args[0], true, args[1], place_in_prompt);
        }
      });
    } else if (name == "embed") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK(1 <= args.size() && args.size() <= 2);
        if (args.size() == 1) {
          // args: inp (with place_in_prompt = kAll)
          *rv = GetChat()->EmbedStep(args[0]);
        } else if (args.size() == 2) {
          // args: inp, place_in_prompt
          PlaceInPrompt place_in_prompt = static_cast<PlaceInPrompt>(static_cast<int>(args[1]));
          *rv = GetChat()->EmbedStep(args[0], true, place_in_prompt);
        }
      });
    } else if (name == "prefill_with_embed") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK(1 <= args.size() && args.size() <= 2);
        if (args.size() == 1) {
          // args: embedding (with decode_next_token = true)
          GetChat()->PrefillWithEmbedStep(args[0]);
        } else if (args.size() == 2) {
          // args: embedding, decode_next_token
          GetChat()->PrefillWithEmbedStep(args[0], args[1]);
        }
      });
    } else if (name == "decode") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { GetChat()->DecodeStep(); });
    // } else if (name == "init_chat_legacy") {
    //   // TODO: remove the legacy initialization func after updating app and web sides.
    //   return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
    //     ICHECK_EQ(args.size(), 5);
    //     GetChat()->InitChatLegacy(args[0], args[1], args[2], args[3], args[4]);
    //   });
    } else if (name == "reset_chat") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 0);
        GetChat()->ResetChat();
      });
    } else if (name == "load_json_override") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 2);
        std::string config_str = args[0];
        bool partial_update = args[1];
        GetChat()->LoadJSONOverride(config_str, partial_update);
      });
    } else if (name == "get_role0") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetChat()->conversation_.roles[0];
      });
    } else if (name == "get_role1") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetChat()->conversation_.roles[1];
      });
    } else if (name == "stopped") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { *rv = GetChat()->Stopped(); });
    } else if (name == "get_message") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { *rv = GetChat()->GetMessage(); });
    } else if (name == "runtime_stats_text") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetChat()->RuntimeStatsText();
      });
    } else if (name == "reset_runtime_stats") {
      return PackedFunc(
          [this, sptr_to_self](TVMArgs args, TVMRetValue* rv) { GetChat()->ResetRuntimeStats(); });
    } else if (name == "get_config_json") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        *rv = GetChat()->GetConfigJSON();
      });
    } else if (name == "process_system_prompts") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        GetChat()->ProcessSystemPrompts();
      });
    } else {
      return PackedFunc(nullptr);
    }
  }

  void Init(DLDevice device) { device_ = device; }

  LLMChat* GetChat() {
    ICHECK(chat_ != nullptr) << "Chat is not initialized via reload";
    return chat_.get();
  }

  const char* type_key() const final { return "mlc.llm_chat"; }

 private:
  std::unique_ptr<LLMChat> chat_ = nullptr;
  DLDevice device_;
};

std::vector<std::string> CountUTF8(const std::string& s) {
  // assume that the string is always valid utf8
  std::vector<std::string> ret;
  for (size_t pos = 0; pos < s.size();) {
    if ((s[pos] & 0x80) == 0x00) {
      ret.push_back(s.substr(pos, 1));
      pos += 1;
    } else if (pos + 1 < s.size() && (s[pos] & 0xE0) == 0xC0 && (s[pos + 1] & 0xC0) == 0x80) {
      ret.push_back(s.substr(pos, 2));
      pos += 2;
    } else if (pos + 1 < s.size() && (s[pos] & 0xF0) == 0xE0 && (s[pos + 1] & 0xC0) == 0x80 &&
               (s[pos + 2] & 0xC0) == 0x80) {
      ret.push_back(s.substr(pos, 3));
      pos += 3;
    } else if (pos + 2 < s.size() && (s[pos] & 0xF8) == 0xF0 && (s[pos + 1] & 0xC0) == 0x80 &&
               (s[pos + 2] & 0xC0) == 0x80 && (s[pos + 3] & 0xC0) == 0x80) {
      ret.push_back(s.substr(pos, 4));
      pos += 4;
    } else {
      LOG(FATAL) << "Invalid UTF8 string";
    }
  }
  return std::move(ret);
}

/*!
 * \brief Get the diff of new message and current message (the delta message).
 * \param curr_message The current message.
 * \param new_message The new message
 * \return The delta message.
 * \note The main complication here is that new_msg can be different from previous message, so we
 need to find the diff, delete previous messages that are different, then print it out.
 This logic is only needed for simple stdout.

 For UI apps that can directly update output text we can simply do last_reply.text =
 chat->GetMessage();
 */
std::string GetDeltaMessage(std::string curr_message, std::string new_message) {
  std::vector<std::string> cur_utf8_chars = CountUTF8(curr_message);
  std::vector<std::string> new_utf8_chars = CountUTF8(new_message);
  // Step 1. Find the index of the first UTF8 char that differs
  size_t pos = std::mismatch(cur_utf8_chars.begin(), cur_utf8_chars.end(), new_utf8_chars.begin(),
                             new_utf8_chars.end())
                   .first -
               cur_utf8_chars.begin();
  // Step 2. Delete the previous message since `pos`
  std::string print = "";
  for (size_t j = pos; j < cur_utf8_chars.size(); ++j) {
    print += "\b \b";
  }
  // Step 3. Print the new message since `pos`
  for (size_t j = pos; j < new_utf8_chars.size(); ++j) {
    print += new_utf8_chars[j];
  }
  return print;
}

// register as a system function that can be queried
TVM_REGISTER_GLOBAL("mlc.get_delta_message").set_body_typed(GetDeltaMessage);

tvm::runtime::Module CreateChatModule(DLDevice device) {
  ObjectPtr<LLMChatModule> n = make_object<LLMChatModule>();
  n->Init(device);
  return Module(n);
}

// register as a system function that can be queried
TVM_REGISTER_GLOBAL("mlc.llm_chat_create").set_body_typed([](int device_type, int device_id) {
  return CreateChatModule(DLDevice{static_cast<DLDeviceType>(device_type), device_id});
});

// // TODO: legacy function to be removed
// tvm::runtime::Module CreateChatModuleLegacy(tvm::runtime::Module executable,
//                                             std::unique_ptr<Tokenizer> tokenizer,
//                                             const tvm::runtime::String& param_path,
//                                             DLDevice device) {
//   ObjectPtr<LLMChatModule> n = make_object<LLMChatModule>();
//   n->InitLegacy(executable, std::move(tokenizer), param_path, device);
//   return Module(n);
// }

// // TODO: legacy function to be removed
// tvm::runtime::Module CreateChatModuleLegacy(tvm::runtime::Module executable,
//                                             const tvm::runtime::String& tokenizer_path,
//                                             const tvm::runtime::String& param_path,
//                                             DLDevice device) {
//   // tokenizer stored in single files.
//   return CreateChatModuleLegacy(executable, TokenizerFromPath(tokenizer_path), param_path, device);
// }

// TODO: legacy function to be removed
// register as a system function that can be queried
// TVM_REGISTER_GLOBAL("mlc.llm_chat_create_legacy")
//     .set_body_typed([](tvm::runtime::Module executable, const tvm::runtime::String& tokenizer_path,
//                        const tvm::runtime::String& param_path, int device_type, int device_id) {
//       return CreateChatModuleLegacy(executable, tokenizer_path, param_path,
//                                     DLDevice{static_cast<DLDeviceType>(device_type), device_id});
//     });

}  // namespace llm
}  // namespace mlc
