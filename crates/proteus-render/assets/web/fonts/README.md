# Embedded fonts

The browser page embeds these so that it renders the same offline as online. Both are licensed
under the SIL Open Font License 1.1 (the licence texts are beside them), which allows embedding
and redistribution with the licence.

| file | family | source | subset |
|---|---|---|---|
| `GeistMono-latin.woff2` | Geist Mono (variable weight) | Google Fonts, `geistmono/v6` | latin (U+0000–00FF and the usual latin punctuation) |
| `Figtree-latin.woff2` | Figtree (variable weight) | Google Fonts, `figtree/v9` | latin |

They stand in for Goga and Freigeist, the s1re.sh text and display faces, whose licences for
embedding in distributed software are not established (see
`docs/design/2026-09-23-proteus-identity-design.md`). Glyphs outside the subset (φ, ψ, ≥) fall
back to the system's fonts.
