// Inkling 2 adapter: registers the architecture, validates and describes the file from its `inkling2_loop.*` metadata, and
// loads the pack into an Inkling2Model.
#include "models/inkling2x/adapter.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/log.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/program/options.hpp"
#include "models/inkling2x/meta.hpp"
#include "models/inkling2x/model.hpp"

#include <cstdlib>
#include <optional>
#include <string>

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

    bool parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o, int& code) const override {
        (void) code;
        if (a != "--spec") return false;
        o.spec = std::atoi(args.next("--spec"));
        o.spec_given = true;
        return true;
    }

    // The pack's DSpark drafter offers its whole block (the anchor and the drafts) unless --spec caps it; a pack without one
    // decodes plainly.
    std::optional<int> finish_flags(program::CommonOptions& o) const override {
        core::ModelFile file;
        std::string err;
        int block = 0;
        if (!o.model.empty() && file.open(o.model, err) && file.find("dspark/gguf") != nullptr && file.load_meta(err))
            if (const core::Json* d = file.meta().get("dspark"))
                if (const core::Json* c = d->get("config"))
                    if (const core::Json* b = c->get("block_size")) block = (int) b->num();
        if (!o.spec_given) o.spec = block;
        if (o.spec >= 2 && block == 0) {
            basalt::log::warn("basalt inkling2x", "--spec %d off: the pack has no dspark/gguf section", o.spec);
            o.spec = 0;
        }
        if (block > 0 && o.spec > block) {
            basalt::log::warn("basalt inkling2x", "--spec %d: the drafter's block holds %d rows", o.spec, block);
            o.spec = block;
        }
        return std::nullopt;
    }
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
