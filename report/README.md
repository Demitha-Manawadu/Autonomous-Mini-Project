# Formatted project report

- `Manawadu_220381M_KF_Report.pdf`: report compiled with LaTeX for reading and printing.
- `Manawadu_220381M_KF_Report.tex`: editable LaTeX source; upload it and the required `assets/` images to Overleaf, or compile locally.
- `Manawadu_220381M_KF_Report.docx`: editable Word version. Pagination may vary by Word version.
- `cover_details.json`: confirmed student and module details.
- `report_evidence.json`: regenerated metrics for the figures and tables.
- `assets/`: official emblem, result figures and layout previews.

The PDF uses Computer Modern fonts, numbered mathematical equations, booktabs tables, labelled figures and a thin black page border. It starts with an introduction and has no abstract, report guide or references section. The cover retains the requested university, faculty and department below the emblem. Word is supplied as an editable companion; the LaTeX PDF is the final formatted version.

To rebuild from the repository root:

```powershell
python -m pip install -r requirements.txt -r report/requirements-report.txt
python report/build_report.py
```

The builder uses Times New Roman installed in `C:/Windows/Fonts` for Word, and the Tectonic engine in `report/tools/tectonic.exe` for the final PDF. This engine was downloaded from the official Tectonic project release (https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0). Its first run downloads TeX packages. The executable and cache files are excluded from Git. On another computer, install Tectonic at that path or compile the `.tex` file using a standard LaTeX installation.

The builder reruns synthetic experiments, including 20 seeds per road scenario and a process-noise tuning study. It does not change the simulation code. The official emblem was downloaded from https://www.uom.lk/assets/images/Uni_emblem.jpg, linked from https://www.uom.lk/emblem-university.
