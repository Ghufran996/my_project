"""Figure 1 (conceptual schematic, no measured data): plant and gas train, restart state machine, hydrogen fate accounting.

Builds Figure1.pptx: one 13.333 x 7.5 in slide, one group of native PowerPoint objects (autoshapes, connectors, text frames,
gradient fills, bevels), no pictures. All objects are built in design coordinates (inches); the group is then scaled to 50 % and centred. Fonts are set for the FINAL size.

    python make_figure1.py [--check] [output.pptx]       (default output: Figure1.pptx next to this script)

--check runs the scripted checks (counts, text fit with Arial metrics, overlap, line clearance), prints the result and exits with 1 on a violation.
Requires python-pptx and Pillow (not dependencies of the h2su analysis code; used only for this drawing).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from lxml import etree
from PIL import ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

A = "http://schemas.openxmlformats.org/drawingml/2006/main"
COL = dict(blue="0072B2", sky="56B4E9", green="009E73", orange="E69F00", purple="CC79A7", black="000000", dgray="595959", mgray="BBBBBB", band="F4F4F4", white="FFFFFF")
C = COL
SCALE = 0.5
INSET = 0.02            # design inches; PowerPoint does not scale text insets with a group, so the usable width is judged with 2 x 0.02 in FINAL inches
FONTS = {(False, False): "C:/Windows/Fonts/arial.ttf", (True, False): "C:/Windows/Fonts/arialbd.ttf", (False, True): "C:/Windows/Fonts/ariali.ttf", (True, True): "C:/Windows/Fonts/arialbi.ttf"}

# Bounded moves of the check step (part (b) check 4): id -> (dx, dy, dw) in design inches, each |value| <= 0.15. Listed in the log.
MOVES: dict[str, tuple[float, float, float]] = {"B2a": (-0.05, 0.0, 0.10), "lab_vent_inert": (0.0, 0.0, 0.10)}
DYC = 0.15          # band C content moved down by 0.15 in design (check 4) so that the bus (y 5.52 in the table) clears the band label C; see the log


def light(hexc: str, f: float = 0.5) -> str:
    r, g, b = (int(hexc[i:i + 2], 16) for i in (0, 2, 4))
    return "".join(f"{int(round(c + (255 - c) * f)):02X}" for c in (r, g, b))


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])
grp = slide.shapes.add_group_shape()
G = grp.shapes
OBJ: dict[str, dict] = {}        # registry for the checks (design coordinates)
SEGS: list[dict] = []
E = Inches


def parse_label(label: str, wrap_break: bool = True):
    """'/' = paragraph break (not printed), '_{...}' = subscript run. Returns a list of paragraphs, each a list of (text, is_sub)."""
    paras = label.split("/") if wrap_break else [label]
    out = []
    for line in paras:
        line = line.strip() if wrap_break else line
        runs = []
        for part in re.split(r"(_\{[^}]*\})", line):
            if not part:
                continue
            if part.startswith("_{"):
                runs.append((part[2:-1], True))
            else:
                runs.append((part, False))
        out.append(runs)
    return out


def set_text(shape, paras, size, color="000000", bold=False, italic=False, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, wrap=False):
    tf = shape.text_frame
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(INSET)
    tf.margin_top = tf.margin_bottom = 0
    for k, runs in enumerate(paras):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = align
        for text, sub in runs:
            r = p.add_run()
            r.text = text
            r.font.size = Pt(size)
            r.font.name = "Arial"
            r.font.bold = bold
            r.font.italic = italic
            r.font.color.rgb = RGBColor.from_string(color)
            if sub:
                r._r.get_or_add_rPr().set("baseline", "-25000")


def grad_fill(shape, hexc):
    spPr = shape._element.spPr
    for tag in ("a:solidFill", "a:gradFill", "a:noFill"):
        for e in spPr.findall(qn(tag)):
            spPr.remove(e)
    gf = etree.fromstring(
        f'<a:gradFill xmlns:a="{A}" rotWithShape="1"><a:gsLst><a:gs pos="0"><a:srgbClr val="{light(hexc)}"/></a:gs><a:gs pos="100000"><a:srgbClr val="{hexc}"/></a:gs></a:gsLst>'
        f'<a:lin ang="5400000" scaled="0"/></a:gradFill>')
    spPr.find(qn("a:prstGeom")).addnext(gf)


def bevel(shape):
    spPr = shape._element.spPr
    spPr.append(etree.fromstring(f'<a:scene3d xmlns:a="{A}"><a:camera prst="orthographicFront"/><a:lightRig rig="threePt" dir="t"/></a:scene3d>'))
    spPr.append(etree.fromstring(f'<a:sp3d xmlns:a="{A}" prstMaterial="matte"><a:bevelT w="38100" h="38100"/></a:sp3d>'))


def moved(sid, x, y, w):
    dx, dy, dw = MOVES.get(sid, (0, 0, 0))
    assert max(abs(dx), abs(dy), abs(dw)) <= 0.15 + 1e-9, sid
    return x + dx, y + dy, w + dw


def box(sid, kind, x, y, w, h, fill=None, label="", text_color="000000", line=None, line_w=0.75, grad=True, bev=False, size=8, bold=False, anchor=MSO_ANCHOR.MIDDLE,
        align=PP_ALIGN.CENTER, wrap=False, adj=None, role="unit", wrap_break=True):
    x, y, w = moved(sid, x, y, w)
    sh = G.add_shape(kind, E(x), E(y), E(w), E(h))
    sh.name = sid
    sh.shadow.inherit = False
    if adj is not None:
        sh.adjustments[0] = adj
    if fill is None:
        sh.fill.background()
    elif grad:
        sh.fill.solid()
        sh.fill.fore_color.rgb = RGBColor.from_string(fill)
        grad_fill(sh, fill)
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = RGBColor.from_string(fill)
    if line:
        sh.line.color.rgb = RGBColor.from_string(line)
        sh.line.width = Pt(line_w)
    else:
        sh.line.fill.background()
    paras = parse_label(label, wrap_break) if label else []
    if label:
        set_text(sh, paras, size, color=text_color, bold=bold, align=align, anchor=anchor, wrap=wrap)
    if bev:
        bevel(sh)
    OBJ[sid] = dict(id=sid, kind=kind, x=x, y=y, w=w, h=h, paras=paras, size=size, bold=bold, italic=False, wrap=wrap, role=role, adj=adj, owner=None, is_text=bool(label))
    return sh


def tbox(sid, x, y, w, h, label, size=7, align=PP_ALIGN.CENTER, bold=False, italic=False, anchor=MSO_ANCHOR.MIDDLE, owner=None):
    x, y, w = moved(sid, x, y, w)
    tb = G.add_textbox(E(x), E(y), E(w), E(h))
    tb.name = sid
    paras = parse_label(label)
    set_text(tb, paras, size, bold=bold, italic=italic, align=align, anchor=anchor, wrap=False)
    OBJ[sid] = dict(id=sid, kind="textbox", x=x, y=y, w=w, h=h, paras=paras, size=size, bold=bold, italic=italic, wrap=False, role="label", adj=None, owner=owner, is_text=True)
    return tb


# glue indices of connection sites (OOXML preset definitions): rectangle-like t0 l1 b2 r3; can t1 l2 b3 r4; hexagon r0 l3
SITE_IDX = {"rect": dict(t=0, l=1, b=2, r=3), "can": dict(t=1, l=2, b=3, r=4), "hex": dict(r=0, l=3)}


def seg(p0, p1, color="000000", w=1.5, dash=False, arrow=False, name="seg", glue0=None, glue1=None, ends=None):
    """One native straight connector with exact endpoints; glue0/glue1 = (shape id, site index) add the connection reference without moving the endpoint."""
    c = G.add_connector(MSO_CONNECTOR.STRAIGHT, E(p0[0]), E(p0[1]), E(p1[0]), E(p1[1]))
    c.name = name
    c.line.color.rgb = RGBColor.from_string(color)
    c.line.width = Pt(w)
    ln = c._element.spPr.find(qn("a:ln"))
    if dash:
        ln.append(etree.fromstring(f'<a:prstDash xmlns:a="{A}" val="dash"/>'))
    if arrow:
        ln.append(etree.fromstring(f'<a:tailEnd xmlns:a="{A}" type="triangle"/>'))
    cx = c._element.nvCxnSpPr.cNvCxnSpPr
    for tag, g in (("a:stCxn", glue0), ("a:endCxn", glue1)):
        if g:
            sid, idx = g
            el = etree.SubElement(cx, qn(tag))
            el.set("id", str(G._spTree.xpath(f'.//p:cNvPr[@name="{sid}"]')[0].get("id")))
            el.set("idx", str(idx))
    SEGS.append(dict(name=name, p0=p0, p1=p1, w=w, dash=dash, arrow=arrow, ends=tuple(ends) if ends else (glue0[0] if glue0 else None, glue1[0] if glue1 else None)))
    return c


def route(pts, color="000000", w=1.5, dash=False, name="route", glue0=None, glue1=None):
    for i in range(len(pts) - 1):
        last = i == len(pts) - 2
        seg(pts[i], pts[i + 1], color, w, dash, arrow=last, name=f"{name}_{i + 1}", glue0=glue0 if i == 0 else None, glue1=glue1 if last else None)


R_, CAN, HEX = MSO_SHAPE.ROUNDED_RECTANGLE, MSO_SHAPE.CAN, MSO_SHAPE.HEXAGON

# ================================================================ bands (sent to back: added first)
for sid, y, h, lab in (("BA", 0.15, 2.40, "A Plant and gas train"), ("BB", 2.62, 2.68, "B Restart state machine"), ("BC", 5.38, 1.97, "C Hydrogen fate accounting")):
    box(sid, R_, 0.20, y, 12.93, h, fill=C["band"], grad=False, role="band", adj=0.03)
    tbox(sid + "_label", 0.30, y + 0.00, 3.40 if sid == "BC" else 3.10, 0.27, lab, size=8, bold=True, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, owner=sid)

# ================================================================ band A
box("A1", MSO_SHAPE.PARALLELOGRAM, 0.40, 0.80, 1.40, 0.80, C["blue"], bev=True)
tbox("A1_label", 0.40, 1.68, 1.40, 0.54, "PV/1.5 MWp", size=8, owner="A1")
box("A2", R_, 2.15, 0.90, 1.00, 0.60, C["mgray"], bev=True)
tbox("A2_label", 2.00, 1.68, 1.30, 0.54, "DC/coupling", size=8, owner="A2")
box("A3", MSO_SHAPE.CUBE, 3.55, 0.78, 1.35, 0.85, C["blue"], bev=True)
tbox("A3_label", 3.525, 0.20, 1.40, 0.54, "PEM stack/1 MW", size=8, owner="A3")
box("A4", CAN, 5.25, 0.75, 0.95, 0.90, C["sky"], bev=True)
tbox("A4_label", 5.10, 1.70, 1.25, 0.54, "Separator,/cooler", size=8, owner="A4")
box("A5", CAN, 6.70, 0.75, 0.95, 0.90, C["sky"], bev=True)
tbox("A5_label", 6.575, 1.70, 1.20, 0.28, "Deoxo", size=8, owner="A5")
box("A6", CAN, 8.15, 0.75, 0.95, 0.90, C["sky"], bev=True)
tbox("A6_label", 8.025, 1.70, 1.20, 0.28, "Dryer", size=8, owner="A6")
box("A7", HEX, 9.55, 0.80, 1.15, 0.80, C["mgray"], bev=True)
tbox("A7_label", 9.475, 1.68, 1.30, 0.54, "Grade D/check", size=8, owner="A7")
box("A8", MSO_SHAPE.PENTAGON, 11.05, 0.85, 1.75, 0.70, C["green"], "Delivered H_{2}", bev=True, adj=0.30)
box("A9", MSO_SHAPE.RECTANGLE, 3.40, 2.16, 1.65, 0.34, C["white"], "Auxiliary supply", line=C["mgray"], grad=False)
box("A10", MSO_SHAPE.RECTANGLE, 5.10, 0.18, 0.60, 0.32, C["white"], "N_{2}", line=C["mgray"], grad=False)
tbox("A11", 6.525, 2.10, 1.30, 0.26, "Gas inventory", size=7, italic=True)

# main line y = 1.20: exact silhouette points of each shape at y = 1.20 (parallelogram edge midpoint, cube and can side edges)
MAIN = [("A1", "A2", (1.70, 1.20), (2.15, 1.20), None, ("A2", 1)),
        ("A2", "A3", (3.15, 1.20), (3.55, 1.20), ("A2", 3), None),
        ("A3", "A4", (4.90, 1.20), (5.25, 1.20), None, ("A4", 2)),
        ("A4", "A5", (6.20, 1.20), (6.70, 1.20), ("A4", 4), ("A5", 2)),
        ("A5", "A6", (7.65, 1.20), (8.15, 1.20), ("A5", 4), ("A6", 2)),
        ("A6", "A7", (9.10, 1.20), (9.55, 1.20), ("A6", 4), ("A7", 3)),
        ("A7", "A8", (10.70, 1.20), (11.05, 1.20), ("A7", 0), ("A8", 1))]
for a, b, p0, p1, g0, g1 in MAIN:
    seg(p0, p1, C["black"], 2.0, arrow=True, name=f"main_{a}_{b}", glue0=g0, glue1=g1, ends=(a, b))
seg((4.225, 2.16), (4.225, 1.63), C["dgray"], 1.0, dash=True, arrow=True, name="feed_aux", glue0=("A9", 0))
seg((5.40, 0.50), (5.40, 0.75), C["dgray"], 1.0, dash=True, arrow=True, name="feed_N2", glue0=("A10", 2))
seg((6.05, 0.75), (6.05, 0.45), C["black"], 1.25, arrow=True, name="vent_blowdown")
tbox("vent_blowdown_label", 6.12, 0.33, 1.70, 0.24, "Blow-down, purge", size=7, align=PP_ALIGN.LEFT)
seg((8.62, 0.75), (8.62, 0.45), C["sky"], 1.25, arrow=True, name="vent_dryer")
tbox("vent_dryer_label", 8.70, 0.33, 1.15, 0.24, "Dryer purge", size=7, align=PP_ALIGN.LEFT)
seg((10.125, 0.80), (10.125, 0.45), C["purple"], 1.25, arrow=True, name="vent_offspec")
tbox("vent_offspec_label", 10.20, 0.33, 1.80, 0.24, "Off-spec diversion", size=7, align=PP_ALIGN.LEFT)

# ================================================================ band B
box("B1", R_, 0.35, 3.25, 0.80, 0.70, C["mgray"], "OFF")
box("B2", R_, 2.20, 3.00, 5.21, 1.10, C["white"], line=C["blue"], line_w=1.5, grad=False, role="container", adj=0.08)
tbox("B2_label", 2.28, 3.03, 0.90, 0.27, "START", size=8, bold=True, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, owner="B2")
for sid, x, w, fc, lab in (("B2a", 2.30, 1.20, light(C["orange"]), "Inerting,/pre-start"), ("B2b", 3.42, 1.00, C["orange"], "Purge"), ("B2c", 4.34, 1.85, C["sky"], "Pressurization"),
                           ("B2d", 6.11, 1.25, C["purple"], "Off-spec")):
    box(sid, MSO_SHAPE.CHEVRON, x, 3.40, w, 0.65, fc, lab, adj=0.25, role="chevron")
box("B3", R_, 8.31, 3.25, 1.60, 0.70, C["green"], "PRODUCTION")
box("B4", R_, 10.91, 3.25, 1.85, 0.70, C["dgray"], "SHUTDOWN", text_color="FFFFFF")
box("B5", R_, 4.20, 4.50, 1.35, 0.40, light(C["sky"]), "STANDBY")

seg((1.15, 3.60), (2.20, 3.60), C["black"], 1.5, arrow=True, name="st_OFF_START", glue0=("B1", 3), glue1=("B2", 1))
tbox("lab_OFF_START", 1.17, 3.30, 0.95, 0.24, "P ≥ P_{min}", size=7)
seg((7.41, 3.60), (8.31, 3.60), C["black"], 1.5, arrow=True, name="st_START_PROD", glue0=("B2", 3), glue1=("B3", 1))
tbox("lab_START_PROD", 7.43, 3.08, 0.86, 0.48, "washout/done", size=7)
seg((9.91, 3.60), (10.91, 3.60), C["black"], 1.5, arrow=True, name="st_PROD_SHUT", glue0=("B3", 3), glue1=("B4", 1))
tbox("lab_PROD_SHUT", 9.93, 3.08, 0.96, 0.48, "Full/shutdown", size=7)
route([(9.11, 3.95), (9.11, 4.70), (5.55, 4.70)], C["black"], 1.5, name="st_PROD_STBY", glue0=("B3", 2), glue1=("B5", 3))
tbox("lab_PROD_STBY", 6.35, 4.42, 1.80, 0.24, "Standby policies", size=7)
seg((4.875, 4.50), (4.875, 4.10), C["black"], 1.5, arrow=True, name="st_STBY_START", glue0=("B5", 0), glue1=("B2", 2))
tbox("lab_STBY_START", 4.95, 4.17, 0.90, 0.24, "P ≥ P_{min}", size=7)
route([(4.875, 4.90), (4.875, 4.97), (11.835, 4.97), (11.835, 3.95)], C["black"], 1.5, name="st_STBY_SHUT", glue0=("B5", 2), glue1=("B4", 2))
tbox("lab_STBY_SHUT", 10.00, 4.71, 0.60, 0.24, "limit", size=7)
route([(4.805, 3.00), (4.805, 2.82), (11.835, 2.82), (11.835, 3.25)], C["black"], 1.5, dash=True, name="st_START_SHUT_abort", glue0=("B2", 0), glue1=("B4", 0))
tbox("lab_abort", 9.00, 2.84, 0.60, 0.24, "abort", size=7)
route([(12.76, 3.60), (12.95, 3.60), (12.95, 5.17), (0.75, 5.17), (0.75, 3.95)], C["black"], 1.5, name="st_SHUT_OFF", glue0=("B4", 3), glue1=("B1", 2))
tbox("lab_vent_inert", 1.30, 4.90, 0.90, 0.24, "vent, inert", size=7)

# ================================================================ band C (y values of the table + DYC)
box("C1", R_, 0.50, 5.95 + DYC, 1.30, 0.75, C["blue"], "M_{stack}")
blocks = [("C2", 2.45, C["green"], "M_{del}", "000000"), ("C3", 3.55, C["orange"], "M_{purge}", "000000"), ("C4", 4.65, C["purple"], "M_{off}", "000000"),
          ("C5", 5.75, C["sky"], "M_{deoxo}", "000000"), ("C6", 6.85, C["sky"], "M_{dry}", "000000"), ("C7", 7.95, C["black"], "M_{blow}", "FFFFFF"),
          ("C8", 9.05, C["mgray"], "ΔM_{inv}", "000000")]
for sid, x, fc, lab, tc in blocks:
    box(sid, R_, x, 5.65 + DYC, 0.98, 0.55, fc, lab, text_color=tc)
box("C9", MSO_SHAPE.RECTANGLE, 10.45, 5.65 + DYC, 2.40, 1.10, None, "λ_{SU} = Σ m_{SU,c} / M_{stack}, then m_{SU,c} = window loss − s(L) share", line=C["mgray"], grad=False,
    align=PP_ALIGN.LEFT, wrap=True, wrap_break=False)
BUS_Y, BUS_X0 = 5.52 + DYC, 2.10
c1r = (1.80, 6.325 + DYC)
seg(c1r, (BUS_X0, c1r[1]), C["black"], 1.0, name="bus_1", glue0=("C1", 3))
seg((BUS_X0, c1r[1]), (BUS_X0, BUS_Y), C["black"], 1.0, name="bus_2")
seg((BUS_X0, BUS_Y), (9.54, BUS_Y), C["black"], 1.0, name="bus_3")
for sid, x, *_ in blocks:
    xc = x + 0.49
    seg((xc, BUS_Y), (xc, 5.65 + DYC), C["black"], 1.0, arrow=True, name=f"drop_{sid}", glue1=(sid, 0))
seg((3.55, 6.60 + DYC), (8.93, 6.60 + DYC), C["dgray"], 1.0, name="bracket_line")
seg((3.55, 6.35 + DYC), (3.55, 6.60 + DYC), C["dgray"], 1.0, name="bracket_tick_left")
seg((8.93, 6.35 + DYC), (8.93, 6.60 + DYC), C["dgray"], 1.0, name="bracket_tick_right")
tbox("C11", 3.55, 6.65 + DYC, 5.38, 0.30, "Cycle window W_{c}: stop to end of off-spec", size=7)

# ================================================================ scale the group to 50 % and centre it
xfrm = grp._element.grpSpPr.find(qn("a:xfrm"))
chOff, chExt = xfrm.find(qn("a:chOff")), xfrm.find(qn("a:chExt"))
cx, cy = int(chExt.get("cx")), int(chExt.get("cy"))
ext_w, ext_h = int(cx * SCALE), int(cy * SCALE)
xfrm.find(qn("a:ext")).set("cx", str(ext_w))
xfrm.find(qn("a:ext")).set("cy", str(ext_h))
off_x, off_y = (prs.slide_width - ext_w) // 2, (prs.slide_height - ext_h) // 2
xfrm.find(qn("a:off")).set("x", str(off_x))
xfrm.find(qn("a:off")).set("y", str(off_y))
grp.name = "Figure1_artwork"
CH0 = (int(chOff.get("x")) / 914400, int(chOff.get("y")) / 914400)
OFF = (off_x / 914400, off_y / 914400)


def fin(x, y):
    """design inches -> final slide inches"""
    return OFF[0] + (x - CH0[0]) * SCALE, OFF[1] + (y - CH0[1]) * SCALE


# ================================================================ checks
EXPECTED_LABELS = [
    "A Plant and gas train", "B Restart state machine", "C Hydrogen fate accounting", "PV 1.5 MWp", "DC coupling", "PEM stack 1 MW", "Separator, cooler", "Deoxo", "Dryer",
    "Grade D check", "Delivered H2", "Auxiliary supply", "N2", "Gas inventory", "Blow-down, purge", "Dryer purge", "Off-spec diversion", "OFF", "START", "Inerting, pre-start",
    "Purge", "Pressurization", "Off-spec", "PRODUCTION", "SHUTDOWN", "STANDBY", "P ≥ Pmin", "P ≥ Pmin", "washout done", "Full shutdown", "Standby policies", "limit", "abort",
    "vent, inert", "Mstack", "Mdel", "Mpurge", "Moff", "Mdeoxo", "Mdry", "Mblow", "ΔMinv", "λSU = Σ mSU,c / Mstack, then mSU,c = window loss − s(L) share",
    "Cycle window Wc: stop to end of off-spec"]
_font_cache: dict = {}


def text_width_in(text, size, bold, italic):
    key = (bold, italic)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONTS[key], 1000)
    return _font_cache[key].getlength(text) / 1000 * size / 72


def para_width(runs, size, bold, italic):
    return sum(text_width_in(t, size, bold, italic) for t, _ in runs)      # subscript runs measured at full size (conservative)


def wrapped_lines(runs, size, bold, italic, width):
    words = re.findall(r"\S+\s*", "".join(t for t, _ in runs))
    lines, cur = [], ""
    for w in words:
        if cur and text_width_in((cur + w).rstrip(), size, bold, italic) > width:
            lines.append(cur)
            cur = w
        else:
            cur += w
    lines.append(cur)
    return len(lines)


def usable(o):
    w_fin, h_fin = o["w"] * SCALE, o["h"] * SCALE
    wu = w_fin - 2 * INSET
    if o["kind"] == MSO_SHAPE.CHEVRON:
        wu -= 2 * o["adj"] * min(o["w"], o["h"]) * SCALE
    elif o["kind"] == MSO_SHAPE.PENTAGON:
        wu -= o["adj"] * min(o["w"], o["h"]) * SCALE
    return wu, h_fin


def check():
    viol = []
    print("--- CHECK 1: counts")
    pres = Presentation(str(OUT))
    sl = pres.slides[0]
    top = list(sl.shapes)
    n_pic = sum(1 for s in sl.shapes._spTree.iter(qn("p:pic")))
    sizes = sorted({int(r.get("sz")) / 100 for r in sl.shapes._spTree.iter(qn("a:rPr")) if r.get("sz")})
    print(f"slide {pres.slide_width / 914400:.3f} x {pres.slide_height / 914400:.3f} in; top-level shapes {len(top)} ({top[0].shape_type}); pictures {n_pic}; run sizes {sizes}")
    if not (abs(pres.slide_width / 914400 - 13.333) < 1e-3 and abs(pres.slide_height / 914400 - 7.5) < 1e-3):
        viol.append("slide size")
    if len(top) != 1 or n_pic != 0 or min(sizes) < 7:
        viol.append("top-level group / pictures / font size")
    texts = []
    for o in OBJ.values():
        if o["is_text"]:
            texts.append(" ".join("".join(t for t, _ in p) for p in o["paras"]).replace("_", ""))
    # compare extracted text from the saved file with the expected list
    got = []
    for sp in sl.shapes[0].shapes:
        if sp.has_text_frame and sp.text_frame.text.strip():
            got.append(" ".join(p.text for p in sp.text_frame.paragraphs))
    exp_sorted, got_sorted = sorted(EXPECTED_LABELS), sorted(got)
    if exp_sorted != got_sorted:
        print("label mismatch: only expected", [l for l in exp_sorted if l not in got_sorted], "only built", [l for l in got_sorted if l not in exp_sorted])
        viol.append("labels")
    print(f"labels: {len(got)} built, {len(EXPECTED_LABELS)} expected, {'equal' if exp_sorted == got_sorted else 'DIFFERENT'}")

    print("--- CHECK 2: text fit (Arial metrics, FINAL inches, insets 2 x 0.02 in)")
    n_fit = 0
    for o in OBJ.values():
        if not o["is_text"]:
            continue
        wu, hu = usable(o)
        nl = 0
        for runs in o["paras"]:
            wd = para_width(runs, o["size"], o["bold"], o["italic"])
            if o["wrap"]:
                nl += wrapped_lines(runs, o["size"], o["bold"], o["italic"], wu)
            else:
                nl += 1
                if wd > wu + 1e-9:
                    viol.append(f"fit width {o['id']}: line {wd:.3f} in > usable {wu:.3f} in")
        if nl * 1.2 * o["size"] / 72 > hu + 1e-9:
            viol.append(f"fit height {o['id']}: {nl} lines need {nl * 1.2 * o['size'] / 72:.3f} in > {hu:.3f} in")
        n_fit += 1
    print(f"{n_fit} text frames checked")

    print("--- CHECK 3: overlap and clearance (FINAL inches)")

    def rect(o):
        x0, y0 = fin(o["x"], o["y"])
        return (x0, y0, x0 + o["w"] * SCALE, y0 + o["h"] * SCALE)

    def inter(a, b, pad=0.0):
        return a[0] < b[2] - pad and b[0] < a[2] - pad and a[1] < b[3] - pad and b[1] < a[3] - pad

    labels = [o for o in OBJ.values() if o["role"] == "label"]
    shapes = [o for o in OBJ.values() if o["role"] in ("unit", "chevron")]
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            if inter(rect(a), rect(b)):
                viol.append(f"label-label {a['id']} x {b['id']}")
        for s in shapes:
            if s["id"] != a["owner"] and inter(rect(a), rect(s)):
                viol.append(f"label-shape {a['id']} x {s['id']}")
        # a label box must stay inside its band
    segs = []
    for s in SEGS:
        p0, p1 = fin(*s["p0"]), fin(*s["p1"])
        segs.append((s, p0, p1))

    def seg_hits_rect(p0, p1, r, shrink=1e-6):
        x0, x1 = sorted((p0[0], p1[0]))
        y0, y1 = sorted((p0[1], p1[1]))
        return x0 < r[2] - shrink and x1 > r[0] + shrink and y0 < r[3] - shrink and y1 > r[1] + shrink

    for s, p0, p1 in segs:
        for l in labels:
            if seg_hits_rect(p0, p1, rect(l)):
                viol.append(f"segment {s['name']} through label {l['id']}")
        for sh in shapes:
            if sh["id"] in s["ends"]:
                continue
            r = rect(sh)
            # end shapes without glue (e.g. cube end of a feed): allow touching, test strict interior
            if seg_hits_rect(p0, p1, r, shrink=0.004):
                viol.append(f"segment {s['name']} through shape {sh['id']}")
        for sh in [o for o in OBJ.values() if o["role"] == "container"]:
            pass
    # parallel clearance between centre lines (>= 0.08 in) for overlapping extents
    n_pair = 0
    mn = 9.0
    for i, (s1, a0, a1) in enumerate(segs):
        for s2, b0, b1 in segs[i + 1:]:
            if s1["name"].split("_")[:2] == s2["name"].split("_")[:2] and s1["name"].startswith(("st_", "bus_", "bracket_")):
                pass
            for horiz in (True, False):
                k = 1 if horiz else 0
                if abs(a0[k] - a1[k]) < 1e-9 and abs(b0[k] - b1[k]) < 1e-9:
                    j = 0 if horiz else 1
                    lo1, hi1 = sorted((a0[j], a1[j]))
                    lo2, hi2 = sorted((b0[j], b1[j]))
                    if min(hi1, hi2) - max(lo1, lo2) > 1e-6:
                        d = abs(a0[k] - b0[k])
                        n_pair += 1
                        if d > 1e-6:
                            mn = min(mn, d)
                        if d < 0.08 - 1e-9:
                            viol.append(f"parallel {s1['name']} / {s2['name']}: {d:.3f} in")
    print(f"{len(segs)} segments, {len(labels)} label boxes, {len(shapes)} shapes; parallel overlapping pairs {n_pair}, smallest clearance {mn:.3f} in")
    print("--- RESULT:", "0 violations" if not viol else f"{len(viol)} violations")
    for v in viol:
        print("   ", v)
    print("bounded moves applied (dx, dy, dw design in):", MOVES, "| band C content (C1-C9, bus, bracket, C11) dy =", DYC)
    return viol


arg = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(arg[0]) if arg else Path(__file__).resolve().parent / "Figure1.pptx"
prs.save(str(OUT))
print("saved", OUT, "| group design extent (in):", round(cx / 914400, 3), "x", round(cy / 914400, 3), "scaled:", round(ext_w / 914400, 3), "x", round(ext_h / 914400, 3), "| objects in group:", len(G))
if "--check" in sys.argv:
    sys.exit(1 if check() else 0)
