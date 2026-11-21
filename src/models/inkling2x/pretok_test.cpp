// src/models/inkling2x/pretok_test.cpp - the o200k pre-tokenizer on strings whose pieces Hugging Face's tokenizers library
// gives for the model's tokenizer.json.
#include "models/inkling2x/adapter.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    using basalt::models::inkling2x::pretok_o200k;
    struct Case { std::string text; std::vector<std::string> pieces; };
    const std::vector<Case> cases = {
        // a word splits where lower case turns upper
        {"HelloWorld", {"Hello", "World"}},
        // an upper run keeps no prefix; a lower word after one upper letter
        {"SCREAMING_CASE iPhone", {"SCREAMING", "_CASE", " i", "Phone"}},
        // the case turn in Latin-1
        {"\xc3" "\x89" "coleNormale", {"\xc3" "\x89" "cole", "Normale"}},
        // the case turn in Greek
        {"\xce" "\xa3" "\xce" "\xaf" "\xcf" "\x83" "\xcf" "\x85" "\xcf" "\x86" "\xce" "\xbf" "\xcf" "\x82" "\xce" "\x91" "\xce" "\xb8" "\xce" "\xae" "\xce" "\xbd" "\xce" "\xb1", {"\xce" "\xa3" "\xce" "\xaf" "\xcf" "\x83" "\xcf" "\x85" "\xcf" "\x86" "\xce" "\xbf" "\xcf" "\x82", "\xce" "\x91" "\xce" "\xb8" "\xce" "\xae" "\xce" "\xbd" "\xce" "\xb1"}},
        // a titlecase digraph opens a word
        {"\xc7" "\x85" "eneva", {"\xc7" "\x85" "eneva"}},
        // contractions in either case, long s for s
        {"they'd WE'RE x'" "\xc5" "\xbf", {"they'd", " WE'RE", " x'" "\xc5" "\xbf"}},
        // numbers in threes
        {"x = 12345", {"x", " =", " ", "123", "45"}},
        // a slash opens a punctuation piece
        {"a/b" "\x0a" "/c", {"a", "/b", "\x0a", "/c"}},
        // punctuation takes the newlines after it
        {"end." "\x0a" "\x0a" "Next", {"end", "." "\x0a" "\x0a", "Next"}},
        // a space run gives its last space to the next word
        {"two  spaces ", {"two", " ", " spaces", " "}},
        // letters without case are one word
        {"\xe6" "\x97" "\xa5" "\xe6" "\x9c" "\xac" "\xe8" "\xaa" "\x9e" " text", {"\xe6" "\x97" "\xa5" "\xe6" "\x9c" "\xac" "\xe8" "\xaa" "\x9e", " text"}},
        // a mark after a space prefixes the word it joins
        {" " "\xcc" "\x81" "x", {" " "\xcc" "\x81" "x"}},
        // a no-break space prefixes a word
        {"\xc2" "\xa0" "Hi", {"\xc2" "\xa0" "Hi"}},
    };
    int failures = 0;
    for (const Case& c : cases) {
        std::vector<size_t> ends;
        pretok_o200k(c.text.data(), c.text.size(), ends);
        std::vector<std::string> got;
        size_t from = 0;
        for (size_t e : ends) { got.push_back(c.text.substr(from, e - from)); from = e; }
        const bool ok = got == c.pieces;
        if (!ok) {
            std::printf("FAIL '%s':", c.text.c_str());
            for (const std::string& p : got) std::printf(" [%s]", p.c_str());
            std::printf("\n");
        }
        failures += !ok;
    }
    std::printf(failures ? "FAILED\n" : "all ok\n");
    return failures ? 1 : 0;
}
