// src/models/inkling2x/pretok_o200k.cpp - the o200k pre-tokenizer of Inkling 2: the text split into the pieces BPE merges.
//
// The expression of the model's tokenizer.json, its alternatives tried in turn at each position:
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   \p{N}{1,3}
//    ?[^\s\p{L}\p{N}]+[\r\n/]*
//   \s*[\r\n]+
//   \s+(?!\S)
//   \s+
// matched by hand with the backtracking a regex engine does: an optional prefix is tried with its character first, a
// greedy run gives back characters until what follows it matches. The first letter class (A) is every letter but
// \p{Ll}, plus marks; the second (B) every letter but \p{Lu} and \p{Lt}, plus marks.
#include "models/inkling2x/adapter.hpp"

#include "unicode_flags.hpp"

namespace basalt::models::inkling2x {

using program::tokenizer::cpt_flag_table;
using program::tokenizer::decode_cp;
using program::tokenizer::kLetter;
using program::tokenizer::kLower;
using program::tokenizer::kMark;
using program::tokenizer::kNumber;
using program::tokenizer::kUpper;
using program::tokenizer::kWhitespace;

namespace {

constexpr size_t kFail = (size_t) -1;

struct Text {
    std::vector<uint32_t> cp;
    std::vector<uint16_t> fl;
    size_t n = 0;
    uint32_t at(size_t i) const { return i < n ? cp[i] : 0; }
    uint16_t flags(size_t i) const { return i < n ? fl[i] : (uint16_t) 0; }
    bool newline(size_t i) const { return i < n && (cp[i] == '\r' || cp[i] == '\n'); }
    bool space(size_t i) const { return (flags(i) & kWhitespace) != 0; }
    bool a(size_t i) const {
        const uint16_t f = flags(i);
        return ((f & kLetter) && !(f & kLower)) || (f & kMark);
    }
    bool b(size_t i) const {
        const uint16_t f = flags(i);
        return ((f & kLetter) && !(f & kUpper)) || (f & kMark);
    }
    // [^\r\n\p{L}\p{N}]
    bool prefix(size_t i) const { return i < n && !newline(i) && !(flags(i) & (kLetter | kNumber)); }
    // [^\s\p{L}\p{N}]
    bool other(size_t i) const { return i < n && !(flags(i) & (kWhitespace | kLetter | kNumber)); }
};

// A letter of a contraction, case-folded: ASCII, and U+017F (long s) for s.
uint32_t fold(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    if (c == 0x017F) return 's';
    return c;
}

// (?i:'s|'t|'re|'ve|'m|'ll|'d)? at i: the end of the contraction, or i.
size_t contraction(const Text& t, size_t i) {
    if (t.at(i) != '\'' || i + 1 >= t.n) return i;
    const uint32_t c1 = fold(t.at(i + 1));
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return i + 2;
    if (i + 2 < t.n) {
        const uint32_t c2 = fold(t.at(i + 2));
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) return i + 3;
    }
    return i;
}

// A*B+ from s: A greedy, given back until B starts, then B greedy; kFail when no B can start.
size_t lower_word(const Text& t, size_t s) {
    size_t j = s;
    while (t.a(j)) ++j;
    for (size_t k = j;; --k) {
        if (t.b(k)) {
            size_t e = k;
            while (t.b(e)) ++e;
            return e;
        }
        if (k == s) return kFail;
    }
}

// A+B* from s; kFail when no A starts there.
size_t upper_word(const Text& t, size_t s) {
    size_t j = s;
    while (t.a(j)) ++j;
    if (j == s) return kFail;
    while (t.b(j)) ++j;
    return j;
}

// [^\r\n\p{L}\p{N}]? then `word`, with the prefix first and without it after; then the optional contraction.
size_t with_prefix(const Text& t, size_t pos, size_t (*word)(const Text&, size_t)) {
    size_t e = kFail;
    if (t.prefix(pos)) e = word(t, pos + 1);
    if (e == kFail) e = word(t, pos);
    return e == kFail ? kFail : contraction(t, e);
}

// The end of the piece that starts at pos.
size_t piece(const Text& t, size_t pos) {
    if (size_t e = with_prefix(t, pos, lower_word); e != kFail) return e;
    if (size_t e = with_prefix(t, pos, upper_word); e != kFail) return e;

    // \p{N}{1,3}
    if (t.flags(pos) & kNumber) {
        size_t e = pos + 1;
        while (e < pos + 3 && (t.flags(e) & kNumber)) ++e;
        return e;
    }

    //  ?[^\s\p{L}\p{N}]+[\r\n/]*
    const size_t s = (t.at(pos) == ' ' && t.other(pos + 1)) ? pos + 1 : pos;
    if (t.other(s)) {
        size_t e = s;
        while (t.other(e)) ++e;
        while (t.newline(e) || t.at(e) == '/') ++e;
        return e;
    }

    size_t e = pos, last_newline = kFail;
    while (t.space(e)) {
        if (t.newline(e)) last_newline = e;
        ++e;
    }
    if (last_newline != kFail) return last_newline + 1;   // \s*[\r\n]+
    if (e == t.n && e > pos) return e;                     // \s+(?!\S) at the end of the text
    if (e - pos >= 2) return e - 1;                        // \s+(?!\S) before a non-space
    if (e > pos) return e;                                 // \s+
    return pos + 1;
}

}  // namespace

void pretok_o200k(const char* text, size_t size, std::vector<size_t>& ends) {
    ends.clear();
    const uint8_t* s = (const uint8_t*) text;
    const uint16_t* table = cpt_flag_table();
    Text t;
    std::vector<size_t> off;
    t.cp.reserve(size);
    t.fl.reserve(size);
    off.reserve(size + 1);
    for (size_t i = 0; i < size;) {
        size_t len = 0;
        const uint32_t c = decode_cp(s, size, i, len);
        off.push_back(i);
        t.cp.push_back(c);
        t.fl.push_back(table[c]);
        i += len;
    }
    off.push_back(size);
    t.n = t.cp.size();
    for (size_t pos = 0; pos < t.n;) {
        pos = piece(t, pos);
        ends.push_back(off[pos]);
    }
}

}  // namespace basalt::models::inkling2x
