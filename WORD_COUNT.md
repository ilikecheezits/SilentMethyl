# Word count

Main text: **4,771 words** (including section/subsection headings), leaving 229 below 5,000.

| Section | Words |
|---|---:|
| Introduction | 631 |
| Results | 3,278 |
| Discussion | 862 |
| Main text total | 4,771 |
| Methods, separately counted | 1,621 |
| Abstract, separately counted | 141 |

The main-text scope follows the [Nature Communications article guidance](https://www.nature.com/ncomms/submit/article): Introduction, Results and Discussion; abstract, Methods, references and figure legends are excluded.

Reproducible convention: select the manuscript from `\section{Introduction}` up to `\section{Methods}`, remove figure environments, then process with OpenDetex 2.8.91 (`detex -l -n -r -s`) and count whitespace-separated tokens. LaTeX comments, citations and figure-reference numbers are stripped. Headings remain; inline maths receives a placeholder token. The extracted text in `main_text_for_count.txt` has the same token count, with mathematical placeholders renamed `[math]` for readability. Complex expressions may be counted differently by another word counter; the margin accommodates small variation. This is a transparent editorial count rather than an assertion that every journal counter gives an identical number.

Full figure insertion and restoration of the missing bibliography will increase the PDF length but do not change this main-text count.
