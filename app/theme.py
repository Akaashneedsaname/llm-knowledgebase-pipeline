"""
Visual theme for the Streamlit app: an archive reading room.

Search results are drawn as library catalog cards (ruled card stock, a red
header rule, a typewritten citation). Everything around them stays quiet:
cool archival grey, ink blue, file-folder tabs.
"""

from html import escape

INK = "#1C2A42"
STACK = "#E6EBEF"      # page background, like archive boxes
CARD = "#FCFDFE"       # card stock surfaces
RULE = "#A9C1D6"       # index-card ruling and borders
MARGIN_RED = "#C2333D"  # the one red rule on a catalog card; also keyboard focus
PENCIL = "#5E6B7B"     # secondary text
FOLDER = "#D7E0E7"     # inactive tabs, sidebar

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,500;7..72,600&family=Hanken+Grotesk:wght@400;500;600&family=Courier+Prime&display=swap');

:root {{
  --ink: {INK}; --stack: {STACK}; --card: {CARD}; --rule: {RULE};
  --red: {MARGIN_RED}; --pencil: {PENCIL}; --folder: {FOLDER};
  --serif: 'Literata', Georgia, 'Times New Roman', serif;
  --sans: 'Hanken Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --typewriter: 'Courier Prime', 'Courier New', monospace;
}}

/* Base type */
.stApp {{ font-family: var(--sans); color: var(--ink); }}
.stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea {{ font-family: var(--sans); }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
  font-family: var(--serif); font-weight: 500; color: var(--ink); letter-spacing: -0.01em;
}}
.stApp h3 {{ font-size: 1.45rem; }}

/* Streamlit chrome */
[data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ max-width: 1120px; padding-top: 2.5rem; }}

/* Masthead */
.kb-masthead {{ margin: 0 0 1.75rem; }}
.kb-masthead h1 {{
  font-family: var(--serif); font-weight: 500; font-size: 2.6rem; line-height: 1.1;
  margin: 0 0 .4rem; padding: 0;
}}
.kb-masthead p {{ color: var(--pencil); font-size: 1.05rem; margin: 0; max-width: 62ch; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: var(--folder); border-right: 1px solid var(--rule); }}
.stApp .kb-status {{ display: flex; align-items: center; gap: .55rem; margin: .15rem 0 .55rem; font-size: .95rem; }}
.kb-dot {{ width: .6rem; height: .6rem; border-radius: 50%; flex: none; }}
.kb-dot.on {{ background: #2F7D5B; }}
.kb-dot.off {{ background: var(--rule); }}
.kb-dot.bad {{ background: var(--red); }}
.stApp .kb-side-note {{ color: var(--pencil); font-size: .9rem; line-height: 1.45; margin: .25rem 0 0; }}

/* Folder tabs */
[data-testid="stTabs"] [role="tablist"] {{ gap: .3rem; border-bottom: none; box-shadow: none; }}
[data-testid="stTabs"] .react-aria-SelectionIndicator {{ display: none; }}
[data-testid="stTabs"] [data-testid="stTab"] {{
  background: var(--folder); border: 1px solid var(--rule); border-bottom: none;
  border-radius: 7px 7px 0 0; padding: .6rem 1.2rem; height: auto; margin: 0 0 -1px;
  position: relative; z-index: 1;
}}
[data-testid="stTabs"] [data-testid="stTab"]:hover {{ background: #E2E9EE; }}
[data-testid="stTabs"] [data-testid="stTab"] p {{ font-size: .98rem; font-weight: 500; color: var(--pencil); }}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {{ background: var(--card); z-index: 2; }}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] p {{ color: var(--ink); font-weight: 600; }}
[data-testid="stTabPanel"] {{
  background: var(--card); border: 1px solid var(--rule); border-top: none;
  border-radius: 0 0 8px 8px; padding: 1.75rem 2rem 2rem;
}}

/* Buttons */
.stApp button[data-testid="stBaseButton-primary"] {{
  background: var(--ink); border: 1px solid var(--ink); color: #fff; border-radius: 5px; font-weight: 600;
}}
.stApp button[data-testid="stBaseButton-primary"]:hover {{ background: #2C3E5E; border-color: #2C3E5E; }}
.stApp button[data-testid="stBaseButton-secondary"] {{
  background: var(--card); border: 1px solid var(--ink); color: var(--ink); border-radius: 5px; font-weight: 500;
}}
.stApp button[data-testid="stBaseButton-secondary"]:hover {{ background: #EEF2F5; color: var(--ink); border-color: var(--ink); }}
.stApp button:focus-visible, .stApp input:focus-visible, .stApp [role="tab"]:focus-visible {{
  outline: 2px solid var(--red); outline-offset: 2px;
}}

/* Inputs */
[data-testid="stFileUploaderDropzone"] {{ background: #F4F7F9; border: 1.5px dashed var(--rule); border-radius: 8px; }}
[data-testid="stTextInputRootElement"] {{ border: 1px solid var(--rule); border-radius: 5px; background: #fff; }}
[data-testid="stTextInputRootElement"] input {{ background: #fff; font-size: 1.02rem; }}
[data-testid="stTextInputRootElement"]:focus-within {{ border-color: var(--ink); }}
[data-testid="stExpander"] details {{ border: 1px solid var(--rule); border-radius: 7px; background: var(--card); }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--rule); border-radius: 7px; }}

/* Prose blocks */
.stApp .kb-lede {{ font-family: var(--serif); font-size: 1.2rem; line-height: 1.55; margin: .2rem 0 1rem; max-width: 70ch; }}
.stApp .kb-note {{ color: var(--pencil); font-size: .92rem; margin: .1rem 0 .9rem; }}
.stApp .kb-files {{ margin: .2rem 0 1rem; padding: 0; list-style: none; }}
.stApp .kb-files li {{ padding: .35rem 0; border-bottom: 1px solid #E3EAF0; font-size: .95rem; }}

/* Run report */
.kb-report {{ border: 1px solid var(--rule); border-radius: 8px; padding: 1.2rem 1.4rem; margin: 0 0 1rem; background: #fff; }}
.kb-report h4 {{ font-size: 1.3rem; margin: 0 0 .15rem; padding: 0; }}
.kb-report .kb-byline {{ color: var(--pencil); font-size: .92rem; margin: 0 0 .8rem; }}
.kb-report .kb-facts {{ font-family: var(--serif); font-size: 1.08rem; line-height: 1.6; margin: 0 0 .5rem; }}
.kb-report .kb-meta {{ color: var(--pencil); font-size: .92rem; line-height: 1.5; margin: .2rem 0 0; }}
.kb-report .kb-meta b {{ color: var(--ink); font-weight: 600; }}

/* Catalog cards */
.kb-card {{
  position: relative; background: #fff; border: 1px solid #C9D7E3; border-radius: 3px;
  padding: .75rem 1.4rem 2.1rem; margin: 0 0 1.1rem; max-width: 54rem;
  box-shadow: 0 1px 0 #DCE5EC, 0 3px 0 -1px #fff, 0 4px 0 -1px #C9D7E3;
}}
.kb-card-head {{
  display: flex; justify-content: space-between; align-items: baseline; gap: 1rem;
  font-family: var(--typewriter); font-size: .95rem; color: var(--ink);
  padding-bottom: .45rem; border-bottom: 2px solid var(--red); margin-bottom: .3rem;
}}
.kb-card-head .kb-rank {{ margin-right: .6rem; }}
.kb-card-head .kb-score {{ color: var(--pencil); white-space: nowrap; font-size: .85rem; }}
.kb-card-body {{
  font-family: var(--serif); font-size: 1.02rem; line-height: 1.75rem; color: #24324A;
  background-image: repeating-linear-gradient(to bottom, transparent 0, transparent calc(1.75rem - 1px), #DCE7F0 calc(1.75rem - 1px), #DCE7F0 1.75rem);
  background-position: 0 .1rem;
}}
.kb-card::after {{
  content: ""; position: absolute; left: 50%; bottom: .6rem; width: .7rem; height: .7rem;
  margin-left: -.35rem; border-radius: 50%; background: var(--stack); box-shadow: inset 0 1px 1px rgba(28,42,66,.25);
}}
.kb-card.compact {{ padding-bottom: .9rem; }}
.kb-card.compact .kb-card-head {{ border-bottom: none; padding-bottom: 0; margin: 0; }}
.kb-card.compact::after {{ display: none; }}

/* Answer */
.kb-answer {{ font-family: var(--serif); font-size: 1.15rem; line-height: 1.7; max-width: 72ch; margin: .2rem 0 1.4rem; }}

@media (max-width: 640px) {{
  [data-testid="stTabPanel"] {{ padding: 1.1rem 1rem 1.4rem; }}
  .kb-masthead h1 {{ font-size: 2rem; }}
  .kb-card {{ padding: .7rem .9rem 2rem; }}
  .kb-card-head {{ flex-direction: column; gap: .2rem; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


def text_to_html(text: str) -> str:
    """
    Escape text and turn line breaks into <br>. Raw blank lines would end the
    HTML block inside st.markdown, and the rest would render as Markdown.
    """
    return escape(text.strip()).replace("\n\n", "<br><br>").replace("\n", "<br>")


def masthead(title: str, subtitle: str) -> str:
    return f'<header class="kb-masthead"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></header>'


def status_row(text: str, state: str) -> str:
    """state: on | off | bad"""
    return f'<div class="kb-status"><span class="kb-dot {state}"></span><span>{escape(text)}</span></div>'


def catalog_card(citation: str, text: str = None, rank: int = None, similarity: float = None) -> str:
    """A library catalog card. Without text, a compact card with only the typed heading."""
    rank_html = f'<span class="kb-rank">[{rank}]</span>' if rank is not None else ""
    score_html = f'<span class="kb-score">similarity {similarity:.2f}</span>' if similarity is not None else ""
    head = f'<div class="kb-card-head"><span>{rank_html}{escape(citation)}</span>{score_html}</div>'
    if text is None:
        return f'<div class="kb-card compact">{head}</div>'
    return f'<div class="kb-card">{head}<div class="kb-card-body">{text_to_html(text)}</div></div>'
