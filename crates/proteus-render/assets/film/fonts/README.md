# Fonts burned into film frames

`proteus film` draws its captions into the pixels with `fontdue`, so it needs the fonts as
TrueType rather than the browser page's WOFF2. Both files are Geist Mono 1.200 (SIL Open Font
License 1.1; the licence text is `Geist-OFL.txt` beside them), from the `geist` npm package
1.3.1 (`dist/fonts/geist-mono/`), unmodified.

| file | weight | used for |
|---|---|---|
| `GeistMono-Regular.ttf` | 400 | captions, numbers |
| `GeistMono-SemiBold.ttf` | 600 | titles, the site notation |
