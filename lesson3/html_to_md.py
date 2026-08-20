"""Convert the eCFR enhanced-HTML rendering of 44 CFR Part 61 to Markdown.

Only used by fetch_corpus.sh, to give the corpus a second file format without
duplicating a document that is already in it as a PDF.
"""
import re
import sys
from html.parser import HTMLParser

SKIP_TAGS = {"script", "style", "button", "nav"}
HEADING = re.compile(r"h([1-6])")


class ECFRToMarkdown(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.skip_depth = 0
        self.heading_level = None

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip_depth += 1
            return
        m = HEADING.fullmatch(tag)
        if m:
            self.heading_level = int(m.group(1))
            self.out.append(f"\n\n{'#' * self.heading_level} ")
        elif tag in ("p", "div", "li", "tr", "br"):
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if HEADING.fullmatch(tag):
            self.heading_level = None
            self.out.append("\n")

    def handle_data(self, data):
        if self.skip_depth:
            return
        text = re.sub(r"[ \t\r\f\v]+", " ", data)
        if text.strip():
            self.out.append(text)

    def markdown(self):
        text = "".join(self.out)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() + "\n"


def main(src, dst):
    with open(src, encoding="utf-8") as f:
        html = f.read()
    parser = ECFRToMarkdown()
    parser.feed(html)
    body = parser.markdown()
    header = (
        "# NFIP Standard Flood Insurance Policy — 44 CFR Part 61\n\n"
        "Source: eCFR, Title 44 Part 61 (National Flood Insurance Program insurance "
        "coverage and rates), including the Standard Flood Insurance Policy forms in "
        "Appendix A: Dwelling Form, General Property Form and Residential Condominium "
        "Building Association Policy.\n\n---\n\n"
    )
    with open(dst, "w", encoding="utf-8") as f:
        f.write(header + body)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
