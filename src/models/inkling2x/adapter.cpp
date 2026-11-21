// Inkling 2 adapter: registers the architecture, validates and describes the file from its `inkling2_loop.*` metadata, and
// loads the pack into an Inkling2Model.
#include "models/inkling2x/adapter.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "models/inkling2x/meta.hpp"
#include "models/inkling2x/model.hpp"

namespace basalt::models::inkling2x {
namespace {

class Adapter final : public model::ModelAdapter {
public:
    explicit Adapter(const char* name) : name_(name) {}
    const char* arch() const override { return name_; }
    std::string validate(const GgufFile& g) const override { return validate_meta(g); }
    model::ModelDesc describe(const GgufFile& g) const override { return describe_meta(g); }
    std::unique_ptr<model::LoadedModel> make_model() const override { return std::make_unique<Inkling2Model>(); }

    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "inkling2"}; }

    // A tool call is a message of its own, <|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{..}}
    // <|end_message|>, opened by a name where a content tag would be: the form models/inkling2's adapter gives the same
    // model and template, which the server's Inkling 2 dialect asks the grammar for.
    program::grammar::ToolForm tool_form() const override {
        program::grammar::ToolForm f;
        f.call_open = "<|message_model|>";
        f.fn_mid = "<|content_invoke_tool_json|>{\"name\":{name},\"args\":";
        f.fn_close = "}";
        f.call_close = "<|end_message|>";
        f.json_args = true;
        f.trigger_pattern = "<\\|message_model\\|>[^<]";
        return f;
    }

    PreTokenizer pretokenizer() const override { return {"inkling2", pretok_o200k}; }
    int max_expert_cards() const override { return 0; }
    const char* model_name() const override { return "Inkling 2 Mini"; }

    // <|content_model_end_sampling|> ends a reply (after its last message). Each message opens with its role's token, so
    // the turn boundary the prompt cache cuts at is the user's (<|message_user|>), which opens no reply. One
    // <|unused_200054|> after <|content_image|> stands for a picture.
    model::TokenNames token_names() const override {
        return {{"<|content_model_end_sampling|>"}, "<|message_user|>", "<|unused_200054|>", "", ""};
    }

private:
    const char* name_;
};

}  // namespace

std::unique_ptr<model::ModelAdapter> make_adapter() {
    // A pack converted from a GGUF keeps that file's `general.architecture` (inkling2_loop) in its gguf_kv, which is what
    // the registry looks an adapter up by; the same adapter answers to it.
    if (model::find_adapter(kGgufArch) == nullptr) model::register_adapter(std::make_unique<Adapter>(kGgufArch));
    return std::make_unique<Adapter>(kArch);
}

}  // namespace basalt::models::inkling2x
