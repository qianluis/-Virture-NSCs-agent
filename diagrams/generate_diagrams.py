#!/usr/bin/env python3
"""
Generate draw.io XML diagrams for VirtualCell-Agent.
Output: .drawio files (editable in diagrams.net) + .png (rendered if possible)
"""
import os, json, xml.etree.ElementTree as ET
from xml.dom import minidom

OUTDIR = os.path.dirname(os.path.abspath(__file__))

def escape(s):
    """XML-escape a string"""
    s = str(s)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = s.replace('"', "&quot;").replace("'", "&apos;")
    return s

def mx_cell(id_, value, style, x=0, y=0, w=100, h=40, vertex=True, src=None, tgt=None, label="",
            entryX=None, entryY=None, exitX=None, exitY=None):
    """Create a draw.io cell element"""
    cell = ET.Element("mxCell", id=id_, value=escape(value), style=style, parent="1")
    if vertex:
        cell.set("vertex", "1")
        geo = ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), as_="geometry")
    else:
        cell.set("edge", "1")
        cell.set("source", src)
        cell.set("target", tgt)
        geo = ET.SubElement(cell, "mxGeometry", relative="1", as_="geometry")
        if label:
            label_elem = ET.SubElement(geo, "mxPoint", x="0", y="-15", as_="offset")
            cell.set("value", escape(label))
        # Add waypoints for nicer routing
        if entryX is not None:
            cell.set("entryX", str(entryX))
            cell.set("entryY", str(entryY))
        if exitX is not None:
            cell.set("exitX", str(exitX))
            cell.set("exitY", str(exitY))
    return cell

def build_mxfile(diagram_name, cells, w=1600, h=1200):
    """Build a complete draw.io XML document"""
    graph_attrs = {
        "dx": "0", "dy": "0", "grid": "1", "gridSize": "10",
        "guides": "1", "tooltips": "1", "connect": "1",
        "arrows": "1", "fold": "1", "page": "1",
        "pageScale": "1", "pageWidth": str(w), "pageHeight": str(h),
        "background": "#ffffff"
    }
    root = ET.Element("mxGraphModel", graph_attrs)
    root.set("dx", "0")
    root.set("dy", "0")
    
    # Root cell
    cell0 = ET.SubElement(root, "root")
    c0 = ET.SubElement(cell0, "mxCell", id="0")
    c1 = ET.SubElement(cell0, "mxCell", id="1", parent="0")
    
    for cell in cells:
        cell0.append(cell)
    
    mxfile = ET.Element("mxfile", host="app.diagrams.net")
    diagram = ET.SubElement(mxfile, "diagram", id=os.urandom(8).hex(), name=diagram_name)
    diagram.append(root)
    
    return mxfile

def prettify_xml(elem):
    """Return a pretty-printed XML string"""
    rough = ET.tostring(elem, encoding="unicode")
    try:
        dom = minidom.parseString(rough.encode())
        return dom.toprettyxml(indent="  ")
    except:
        return rough

# ============================================================
# DIAGRAM 1: System Architecture (3-tier)
# ============================================================
def diagram_system_architecture():
    cells = []
    # Colors
    DATA_BG = "#FFF3E0"
    DATA_S = "#EF6C00"
    MODEL_BG = "#D4E6F1"
    MODEL_S = "#2C6E9C"
    FIT_BG = "#FCE4EC"
    FIT_S = "#C62828"
    AI_BG = "#E8EAF6"
    AI_S = "#283593"
    VAL_BG = "#E8F5E9"
    VAL_S = "#388E3C"
    LEGEND_FG = "#666666"
    
    y_start = 20
    box_style = "rounded;html=1;fontSize=11;fontStyle=1;align=center;verticalAlign=middle;whiteSpace=wrap;"
    
    # Title
    cells.append(mx_cell("title", "VirtualCell-Agent: Neural Stem Cell GRN Simulation Platform",
        "text;html=1;fontSize=16;fontStyle=1;align=center;verticalAlign=middle;fillColor=none;strokeColor=none;",
        x=0, y=10, w=1600, h=30, vertex=True))
    
    # === TIER 1: DATA LAYER ===
    cells.append(mx_cell("t1label", "<b>TIER 1: Data Layer</b>",
        "text;html=1;fontSize=13;fontStyle=1;fillColor=#F5F5F5;strokeColor=#BDBDBD;rounded=1;align=center;",
        x=40, y=50, w=1520, h=28, vertex=True))
    
    # Data source boxes
    ds = [
        ("sc",  "🧬 scRNA-seq",          "10X Genomics h5",  50,  90, 220, 65, "#E8F5E9", "#388E3C"),
        ("lit", "📄 Literature",          "PubMed / BioRxiv", 310, 90, 220, 65, "#FFF8E1", "#F57F17"),
        ("ps",  "⚙️ Perturb-seq",         "KO/OE/Drug screens", 570, 90, 220, 65, "#F3E5F5", "#8E24AA"),
        ("sbml","🔬 SBML Models",         "BioModels/PhysiCell", 830, 90, 220, 65, "#E1F5FE", "#0288D1"),
    ]
    for id_, title, desc, x, y, w, h, bg, stroke in ds:
        style_str = f"rounded=1;html=1;fontSize=10;fontStyle=1;align=center;verticalAlign=middle;whiteSpace=wrap;fillColor={bg};strokeColor={stroke};"
        cells.append(mx_cell(id_, f"<b>{title}</b><br><font style='font-size:9px'>{desc}</font>",
            style_str, x=x, y=y, w=w, h=h, vertex=True))
    
    # Data Processor Hub
    data_proc_style = f"rounded=1;html=1;fontSize=10;fillColor=#ECEFF1;strokeColor=#546E7A;fontStyle=1;align=center;verticalAlign=middle;whiteSpace=wrap;"
    cells.append(mx_cell("dp", "<b>⚡ Unified Data Processor</b><br><font style='font-size:9px'>parse_scdata.py / parse_sbml.py<br>search_literature.py</font>",
        data_proc_style, x=1160, y=80, w=280, h=85, vertex=True))
    
    # Arrows: data sources → processor
    edge_style = "edgeStyle=orthogonalEdgeStyle;html=1;rounded=0;"
    for id_ in ["sc", "lit", "ps", "sbml"]:
        cells.append(mx_cell(f"e_{id_}", "", edge_style, src=id_, tgt="dp", w=50, h=50, vertex=False))
    
    # Data proc → Model tier arrow
    cells.append(mx_cell("e_dp_t2", "processed data", "edgeStyle=orthogonalEdgeStyle;html=1;fontSize=9;",
        src="dp", tgt="t2", w=50, h=50, vertex=False))
    
    # === TIER 2: MODEL & SIMULATION ===
    cells.append(mx_cell("t2", "<b>TIER 2: Model &amp; Simulation Layer</b>",
        "text;html=1;fontSize=13;fontStyle=1;fillColor=#F5F5F5;strokeColor=#BDBDBD;rounded=1;align=center;",
        x=40, y=180, w=1520, h=28, vertex=True))
    
    models = [
        ("grn", "🧠<br><b>GRN Engine</b>", "core/grn_model.py<br>22 genes · Hill ODEs<br>Fold-change repression", 50, 220, 260, 100, MODEL_BG, MODEL_S),
        ("vc", "🔬<br><b>VirtualCell Core</b>", "core/virtual_cell.py<br>Perturbation simulator<br>Expression classification", 380, 220, 260, 100, MODEL_BG, MODEL_S),
        ("pf", "📊<br><b>Parameter Fitting</b>", "MCMC · Nelder-Mead<br>Synthetic validation", 710, 220, 260, 100, FIT_BG, FIT_S),
        ("ai", "🤖<br><b>AI Predictor</b>", "DL-based prediction<br>ODE baseline comparison", 1040, 220, 260, 100, AI_BG, AI_S),
    ]
    for id_, title, desc, x, y, w, h, bg, stroke in models:
        style_str = f"rounded=1;html=1;fontSize=10;fontStyle=0;align=center;verticalAlign=middle;whiteSpace=wrap;fillColor={bg};strokeColor={stroke};fontStyle=1;"
        cells.append(mx_cell(id_, f"{title}<br><font style='font-size:9px'>{desc}</font>",
            style_str, x=x, y=y, w=w, h=h, vertex=True))
    
    # Model internal edges
    cells.append(mx_cell("e_grn_vc", "ODE integration", "edgeStyle=orthogonalEdgeStyle;html=1;fontSize=9;", src="grn", tgt="vc", vertex=False))
    cells.append(mx_cell("e_vc_pf", "benchmark data for fitting", "edgeStyle=orthogonalEdgeStyle;html=1;fontSize=9;", src="vc", tgt="pf", vertex=False))
    cells.append(mx_cell("e_vc_ai", "prediction targets", "edgeStyle=orthogonalEdgeStyle;html=1;fontSize=9;", src="vc", tgt="ai", vertex=False))
    
    # === TIER 3: VALIDATION & OUTPUT ===
    cells.append(mx_cell("t3", "<b>TIER 3: Validation &amp; Output Layer</b>",
        "text;html=1;fontSize=13;fontStyle=1;fillColor=#F5F5F5;strokeColor=#BDBDBD;rounded=1;align=center;",
        x=40, y=360, w=1520, h=28, vertex=True))
    
    val_boxes = [
        ("bc", "✅<br><b>Baseline Check</b>", "AI vs ODE comparison<br>Confidence grading", 50, 400, 240, 75, VAL_BG, VAL_S),
        ("bm", "📈<br><b>Benchmark Suite</b>", "10 perturbations × 22 genes<br>RMSE vs literature", 340, 400, 240, 75, VAL_BG, VAL_S),
        ("fg", "🖼️<br><b>Publication Figures</b>", "Heatmap · Toggle · Topology<br>300 DPI, paper-ready", 630, 400, 240, 75, VAL_BG, VAL_S),
        ("op", "📦<br><b>Deliverables</b>", "Expression profiles<br>JSON results · PNG plots", 920, 400, 240, 75, VAL_BG, VAL_S),
    ]
    for id_, title, desc, x, y, w, h, bg, stroke in val_boxes:
        style_str = f"rounded=1;html=1;fontSize=10;fontStyle=1;align=center;verticalAlign=middle;whiteSpace=wrap;fillColor={bg};strokeColor={stroke};"
        cells.append(mx_cell(id_, f"{title}<br><font style='font-size:9px'>{desc}</font>",
            style_str, x=x, y=y, w=w, h=h, vertex=True))
    
    # Edges: Model → Validation
    cells.append(mx_cell("e_ai_bc", "AI predictions", "edgeStyle=orthogonalEdgeStyle;html=1;fontSize=9;", src="ai", tgt="bc", vertex=False))
    cells.append(mx_cell("e_vc_bm", "perturbation results", "edgeStyle=orthogonalEdgeStyle;html=1;fontSize=9;", src="vc", tgt="bm", vertex=False))
    cells.append(mx_cell("e_bc_fg", "validated results", "edgeStyle=orthogonalEdgeStyle;html=1;fontSize=9;", src="bc", tgt="fg", vertex=False))
    cells.append(mx_cell("e_bm_fg", "benchmark data", "edgeStyle=orthogonalEdgeStyle;html=1;fontSize=9;", src="bm", tgt="fg", vertex=False))
    cells.append(mx_cell("e_fg_op", "figures", "edgeStyle=orthogonalEdgeStyle;html=1;fontSize=9;", src="fg", tgt="op", vertex=False))
    
    # Legend
    legend_text = "<b>Legend:</b>&nbsp;"
    legend_items = [
        ("#EF6C00", "Data Source"),
        ("#2C6E9C", "Model/Simulation"),
        ("#C62828", "Fitting"),
        ("#283593", "AI"),
        ("#388E3C", "Validation"),
    ]
    for color, text in legend_items:
        legend_text += f"<font color='{color}'>■</font> {text}&nbsp;&nbsp;&nbsp;"
    cells.append(mx_cell("leg", legend_text,
        "text;html=1;fontSize=11;fillColor=none;strokeColor=none;align=left;",
        x=50, y=510, w=1500, h=24, vertex=True))
    
    return cells, 1600, 600

# ============================================================
# DIAGRAM 2: GRN Topology (gene network with regulation)
# ============================================================
def diagram_grn_topology():
    cells = []
    # Colors per pathway
    path_colors = {
        "stemness": "#1b7837", "notch": "#2166ac", "proneural": "#b2182b",
        "wnt": "#f4a582", "shh": "#fddbc7", "bmp": "#92c5de", "diff": "#67001f"
    }
    
    cells.append(mx_cell("grn_title", "<b>Neural Stem Cell Gene Regulatory Network</b> | 22 genes · Fold-change repression · Hill ODEs",
        "text;html=1;fontSize=15;fontStyle=1;align=center;fillColor=none;strokeColor=none;",
        x=0, y=10, w=1400, h=30, vertex=True))
    
    # === Node positions (x, y, w, h, pathway) ===
    node_list = [
        ("SOX2", 100, 70, 80, 36, "stemness"),
        ("NES", 50, 120, 80, 36, "stemness"),
        ("PROM1", 100, 120, 80, 36, "stemness"),
        ("TERT", 150, 120, 80, 36, "stemness"),
        ("NOTCH1", 50, 180, 80, 36, "notch"),
        ("HES1", 50, 230, 80, 36, "notch"),
        ("HES5", 50, 280, 80, 36, "notch"),
        ("ASCL1", 180, 230, 80, 36, "proneural"),
        ("CTNNB1", 350, 70, 80, 36, "wnt"),
        ("MYC", 350, 120, 80, 36, "wnt"),
        ("CCND1", 350, 170, 80, 36, "wnt"),
        ("NEUROG2", 250, 280, 80, 36, "wnt"),
        ("GLI1", 400, 280, 80, 36, "shh"),
        ("MYCN", 400, 230, 80, 36, "shh"),
        ("SMAD1", 50, 340, 80, 36, "bmp"),
        ("ID1", 50, 390, 80, 36, "bmp"),
        ("GFAP", 180, 390, 80, 36, "bmp"),
        ("DCX", 250, 340, 80, 36, "diff"),
        ("TUBB3", 250, 390, 80, 36, "diff"),
        ("RBFOX3", 350, 340, 80, 36, "diff"),
        ("NEUROD1", 350, 390, 80, 36, "diff"),
        ("MBP", 450, 390, 80, 36, "diff"),
    ]
    
    for name, x, y, w, h, pathway in node_list:
        color = path_colors.get(pathway, "#999999")
        text_color = "white" if pathway in ("stemness", "notch", "proneural", "diff") else "black"
        style = f"rounded=1;html=1;fontSize=9;fontStyle=1;align=center;verticalAlign=middle;fillColor={color};strokeColor=#000000;fontColor={text_color};whiteSpace=wrap;"
        cells.append(mx_cell(f"n_{name}", name, style, x=x, y=y, w=w, h=h, vertex=True))
    
    # === Regulation edges (from core/grn_model.py) ===
    regulations = [
        ("SOX2", "HES1", "act", "#2166ac"),
        ("SOX2", "NES", "act", "#2166ac"),
        ("NOTCH1", "HES1", "act", "#2166ac"),
        ("NOTCH1", "HES5", "act", "#2166ac"),
        ("HES1", "ASCL1", "rep", "#d6604d"),
        ("HES5", "ASCL1", "rep", "#d6604d"),
        ("HES1", "NEUROG2", "rep", "#d6604d"),
        ("HES5", "NEUROG2", "rep", "#d6604d"),
        ("ASCL1", "ASCL1", "act", "#2166ac"),  # auto-activation
        ("ASCL1", "NEUROG2", "act", "#2166ac"),
        ("ASCL1", "DCX", "act", "#2166ac"),
        ("ASCL1", "TUBB3", "act", "#2166ac"),
        ("CTNNB1", "MYC", "act", "#2166ac"),
        ("CTNNB1", "CCND1", "act", "#2166ac"),
        ("MYC", "CCND1", "act", "#2166ac"),
        ("CTNNB1", "SOX2", "act", "#2166ac"),
        ("SOX2", "PROM1", "act", "#2166ac"),
        ("SOX2", "TERT", "act", "#2166ac"),
        ("NOTCH1", "HES1", "act", "#2166ac"),
        ("SMAD1", "ID1", "act", "#2166ac"),
        ("ID1", "GFAP", "act", "#2166ac"),
        ("NEUROG2", "DCX", "act", "#2166ac"),
        ("NEUROG2", "TUBB3", "act", "#2166ac"),
        ("NEUROD1", "MBP", "act", "#2166ac"),
    ]
    
    i = 0
    for src, tgt, rtype, color in regulations:
        if src == tgt:
            # Self-loop
            style = f"html=1;rounded=0;strokeColor={color};strokeWidth=2;loop=1;"
        elif rtype == "rep":
            style = f"edgeStyle=orthogonalEdgeStyle;html=1;rounded=0;strokeColor={color};strokeWidth=2;endArrow=block;endFill=0;dashed=1;"
        else:
            style = f"edgeStyle=orthogonalEdgeStyle;html=1;rounded=0;strokeColor={color};strokeWidth=2;endArrow=block;"
        cells.append(mx_cell(f"e_{i}", "", style, src=f"n_{src}", tgt=f"n_{tgt}", vertex=False))
        i += 1
    
    # Legend
    legend_text = "<b>Pathways:</b> "
    lx = 60
    for pn, pd, pc in [("Stemness", "SOX2/NES/PROM1/TERT", "#1b7837"),
                        ("Notch", "NOTCH1/HES1/HES5", "#2166ac"),
                        ("Proneural", "ASCL1/NEUROG2", "#b2182b"),
                        ("Wnt", "CTNNB1/MYC/CCND1", "#f4a582"),
                        ("SHH", "GLI1/MYCN", "#fddbc7"),
                        ("BMP", "SMAD1/ID1/GFAP", "#92c5de"),
                        ("Diff", "DCX/TUBB3/RBFOX3/NEUROD1/MBP", "#67001f")]:
        text_c = "white" if pc in ("#1b7837", "#2166ac", "#b2182b", "#67001f") else "black"
        cells.append(mx_cell(f"l_{pn}", pn,
            f"rounded=0;html=1;fontSize=9;fontStyle=1;fillColor={pc};fontColor={text_c};strokeColor=none;",
            x=lx, y=450, w=70, h=20, vertex=True))
        cells.append(mx_cell(f"ld_{pn}", pd,
            f"text;html=1;fontSize=8;fillColor=none;strokeColor=none;align=left;",
            x=lx+75, y=450, w=200, h=20, vertex=True))
        lx += 300
    
    # Edge style legend
    cells.append(mx_cell("leg_edge", "─────── Activation  - - - - - - - → Repression  ⤾ Self-activation loop",
        "text;html=1;fontSize=9;fillColor=none;strokeColor=none;align=center;",
        x=300, y=480, w=800, h=20, vertex=True))

    return cells, 1400, 520

# ============================================================
# DIAGRAM 3: MCMC + Benchmark Workflow
# ============================================================
def diagram_workflow():
    cells = []
    
    cells.append(mx_cell("wf_title", "<b>VirtualCell Parameter Estimation &amp; Validation Workflow</b>",
        "text;html=1;fontSize=15;fontStyle=1;align=center;fillColor=none;strokeColor=none;",
        x=0, y=10, w=1400, h=30, vertex=True))
    
    # Flowchart boxes
    steps = [
        ("s0", "📖 Literature Curation", "Andersen 2021 · Imayoshi 2010\nGao 2009 · Zhang 2019", 50, 60, 180, 70, "#FFF3E0", "#EF6C00"),
        ("s1", "🧬 GRN ODE Model", "22 genes, fold-change repression\nHill function dynamics", 300, 60, 180, 70, "#D4E6F1", "#2C6E9C"),
        ("s2", "🔄 Steady-State Solver", "scipy.integrate.odeint\nt=500, rtol=1e-6", 550, 60, 180, 70, "#D4E6F1", "#2C6E9C"),
        ("s3", "⚡ Apply Perturbation", "KO: basal=0, deg×3\nOE: basal×5, deg×0.3", 800, 60, 180, 70, "#D4E6F1", "#2C6E9C"),
        
        ("s4", "📊 Compute log₂FC", "log2(perturbed/control)\nz-score normalization", 50, 170, 180, 70, "#FCE4EC", "#C62828"),
        ("s5", "🎯 Parameter Fitting", "Nelder-Mead (12 params)\nMCMC (44 params, 16×500)", 300, 170, 180, 70, "#FCE4EC", "#C62828"),
        ("s6", "🔬 Synthetic Validation", "Known → perturb → recover\nRMSE / Pearson r", 550, 170, 180, 70, "#FCE4EC", "#C62828"),
        
        ("s7", "📈 Benchmark Suite", "10 perturbations × 22 genes\nHeatmap generation", 50, 280, 180, 70, "#E8F5E9", "#388E3C"),
        ("s8", "🖼️ Paper Figures", "Fig1: Atlas · Fig2: Toggle\nFig3: Topology · Fig4: Recovery", 300, 280, 180, 70, "#E8F5E9", "#388E3C"),
        ("s9", "📝 Report & Deliver", "Methods: §2.3 Parameter Est.\nResults: §3.3 Benchmark", 550, 280, 180, 70, "#E8F5E9", "#388E3C"),
    ]
    
    for id_, title, desc, x, y, w, h, bg, stroke in steps:
        style = f"rounded=1;html=1;fontSize=10;fontStyle=1;align=center;verticalAlign=middle;whiteSpace=wrap;fillColor={bg};strokeColor={stroke};"
        cells.append(mx_cell(id_, f"<b>{title}</b><br><font style='font-size:8px'>{desc}</font>", style, x=x, y=y, w=w, h=h, vertex=True))
    
    # Flow arrows
    edge_style = "edgeStyle=orthogonalEdgeStyle;html=1;rounded=0;strokeWidth=1.5;endArrow=classic;fontSize=9;"
    flows = [
        ("fs0_s1", "targets",  "s0", "s1"),
        ("fs1_s2", "ODE system", "s1", "s2"),
        ("fs2_s3", "steady state", "s2", "s3"),
        ("fs3_s4", "results", "s3", "s4"),
        ("fs4_s5", "log₂FC matrix", "s4", "s5"),
        ("fs5_s6", "best params", "s5", "s6"),
        ("fs6_s7", "validated", "s6", "s7"),
        ("fs7_s8", "data", "s7", "s8"),
        ("fs8_s9", "figures", "s8", "s9"),
        ("fs3b_s7", "perturbation results", "s3", "s7"),
    ]
    for id_, label, src, tgt in flows:
        cells.append(mx_cell(id_, label, edge_style, src=src, tgt=tgt, vertex=False))
    
    # Feedback loop
    cells.append(mx_cell("fb", "feedback (parameter refinement)", 
        "edgeStyle=orthogonalEdgeStyle;html=1;rounded=0;strokeWidth=1.5;endArrow=classic;fontSize=9;dashed=1;strokeColor=#C62828;",
        src="s7", tgt="s5", vertex=False))
    
    cells.append(mx_cell("wf_legend", 
        "<b>Color code:</b> <font color='#EF6C00'>■</font> Curation <font color='#2C6E9C'>■</font> Simulation <font color='#C62828'>■</font> Fitting <font color='#388E3C'>■</font> Validation &amp; Output",
        "text;html=1;fontSize=11;fillColor=none;strokeColor=none;align=center;",
        x=200, y=390, w=1000, h=24, vertex=True))
    
    return cells, 1400, 440

# ============================================================
# Write all diagrams
# ============================================================
diagrams = [
    ("System Architecture", diagram_system_architecture, 1600, 600),
    ("GRN Topology", diagram_grn_topology, 1400, 520),
    ("MCMC Workflow", diagram_workflow, 1400, 440),
]

for name, func, w, h in diagrams:
    print(f"Generating: {name}...", end=" ")
    cells, _, _ = func()
    mxfile = build_mxfile(name, cells, w=w, h=h)
    xml_str = prettify_xml(mxfile)
    filename = f"{name.lower().replace(' ', '_')}.drawio"
    with open(os.path.join(OUTDIR, filename), "w", encoding="utf-8") as f:
        f.write(xml_str)
    size = os.path.getsize(os.path.join(OUTDIR, filename))
    print(f"✅ {size/1024:.0f} KB")

print(f"\n✅ All diagrams saved to: {OUTDIR}")
print("   Open in: https://app.diagrams.net")
