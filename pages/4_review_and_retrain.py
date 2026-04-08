import sys, os

_src = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "Autoclassification Scheme", "streamlit_demo", "streamlit_app", "src",
))
if _src not in sys.path:
    sys.path.insert(0, _src)

_page = os.path.join(_src, "pages", "4_review_and_retrain.py")
exec(compile(open(_page).read(), _page, "exec"))
