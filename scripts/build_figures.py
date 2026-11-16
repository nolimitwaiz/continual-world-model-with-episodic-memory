#!/usr/bin/env python3
"""Deterministic SVG diagrams and figures from stored artifacts; no model execution."""
from pathlib import Path
from statistics import mean
from html import escape
import json

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs/figures"
ORDER = ["context_only", "frame_memory", "fixed_chunks", "random_matched", "oracle_events"]
LABELS = ["Recent history", "Single steps", "Fixed chunks", "Random chunks", "Events (given)"]
COLORS = ["#8796A5", "#657A8E", "#4778BA", "#A76B23", "#007E78"]

def load(relative):
    return json.loads((ROOT / relative).read_text())

def e2_scores():
    scores = {key: {} for key in ORDER}
    for path in (ROOT / "results/cluster/1779882_e2_final").glob("*/seed-*/result.json"):
        result = json.loads(path.read_text())
        scores[result["condition"]][result["seed"]] = result["validation"]["long_delay_memory_score"]
    return {key: [value[s] for s in sorted(value)] for key, value in scores.items()}

class SVG:
    def __init__(self, title, subtitle, height=600, width=1100):
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
                      f'<title>{escape(title)}</title><desc>{escape(subtitle)}</desc>',
                      '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#607083"/></marker></defs>',
                      '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172C43} .small{fill:#526579}</style>',
                      f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>']
        self.text(36, 42, title, 26, weight=700)
        self.text(36, 72, subtitle, 16, color="#526579")
    def text(self, x, y, s, size=18, color="#172C43", weight=400, anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" style="fill:{color}">{escape(str(s))}</text>')
    def line(self, x1, y1, x2, y2, color="#DCE4EA", width=1, arrow=False, dash=False):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"'+(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="6 5"' if dash else '')+'/>')
    def path(self, points, color="#607083", width=2, arrow=False, dash=False):
        self.parts.append('<polyline points="'+" ".join(f"{x},{y}" for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{width}"'+(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="6 5"' if dash else '')+'/>')
    def rect(self, x, y, w, h, fill, stroke="none", dash=False):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
    def circle(self, x, y, r, fill, stroke="none"):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}"/>')
    def box(self, x, y, w, title, detail, proposed=False, accent=False):
        self.rect(x,y,w,84,"#E7F4F2" if accent else "#F0F4F8", "#A7B9C8" if proposed else "none", proposed)
        self.text(x+15,y+31,title,19,weight=700)
        self.text(x+15,y+59,detail,15,color="#526579")
    def save(self, name):
        (FIG / name).write_text("\n".join(self.parts + ["</svg>"]) + "\n")

def architecture():
    p=SVG("A small model with limited memory", "Simplified overview. A later question determines which stored records are read.",265)
    boxes=[(36,"Experience","Symbolic states and actions"),
           (300,"Recent history","A small recurrent network"),
           (564,"Group and store","Up to 64 memory records"),
           (828,"Answer a question","Read up to four records")]
    for i,(x,title,detail) in enumerate(boxes):
        p.box(x,115,235,title,detail,accent=(i==2))
        if i<3:p.line(x+235,157,x+254,157,arrow=True)
    p.text(36,238,"Completed tests supply event boundaries. The full architecture includes a recent-context path to the answer.",16)
    p.save("overview.svg")
    p=SVG("The current memory model", "The experiment changes how experience is grouped into stored records.",590)
    p.box(36,120,245,"Simulated experience","States + observed actions")
    p.box(330,120,260,"Input representation","128 numbers per step")
    p.box(650,120,365,"Recent-history network","GRU; resets between short blocks")
    p.line(281,162,320,162,arrow=True); p.line(590,162,640,162,arrow=True)
    p.box(650,275,365,"Group steps into memories","Individual / fixed / random / event",accent=True)
    p.box(330,275,260,"Stored memories","64 records; oldest leaves first",accent=True)
    p.box(36,275,245,"A later question","Converted to a query vector")
    p.line(832,204,832,265,arrow=True)
    p.line(650,317,600,317,arrow=True)
    p.box(330,430,260,"Read relevant memories","Select up to four records")
    p.box(650,430,365,"Answer the question","Recent history + selected records")
    p.line(460,359,460,420,arrow=True)
    p.path([(158,359),(158,472),(320,472)],arrow=True)
    p.line(590,472,640,472,arrow=True)
    p.path([(1015,162),(1052,162),(1052,410),(832,410),(832,420)],arrow=True)
    p.text(36,556,"Event boundaries are supplied. Inputs are symbolic, not images, speech, or brain activity.",16)
    p.save("architecture.svg")
    p=SVG("Grouping is the intervention", "Schematic example: the same latent stream can be written into different memory records.",465)
    labels=["Individual steps","Fixed chunks","Random chunks","Event aligned"]
    groups=[[1]*12,[3,3,3,3],[2,5,2,3],[3,2,5,2]]
    for row,(label,lengths) in enumerate(zip(labels,groups)):
        y=120+row*75
        p.text(36,y+27,label,19,weight=700)
        cursor=260
        for group,n in enumerate(lengths):
            p.rect(cursor,y,n*60-8,43,"#E7F4F2" if row==3 else "#EDF1F6")
            for index in range(n):
                p.circle(cursor+24+index*60,y+21,6,"#007E78" if row==3 else "#657A8E")
            cursor += n*60
    p.text(36,439,"Illustration only. Matched record payloads do not automatically match retention or source availability.",16)
    p.save("grouping.svg")
    p=SVG("Proposed continual world model", "Research target — dashed modules and connections are not completed-system evidence.",630)
    p.box(36,125,290,"Multimodal perception","Text, speech and visual interfaces",True)
    p.box(400,125,290,"World-state dynamics","Maintain and predict hidden state",True)
    p.box(764,125,300,"Event detection","Learn online boundary decisions",True)
    p.line(326,167,390,167,arrow=True,dash=True);p.line(690,167,754,167,arrow=True,dash=True)
    p.box(764,310,300,"Episodic memory","Write + reactivate related episodes",True,True)
    p.box(400,310,290,"Fast parameter adaptation","Update an explicit fast-weight set",True)
    p.box(36,310,290,"Replay and consolidation","Select experiences for learning",True)
    p.line(914,209,914,300,arrow=True,dash=True)
    p.line(764,352,700,352,arrow=True,dash=True);p.line(400,352,336,352,arrow=True,dash=True)
    p.box(36,480,290,"Slow parameter learning","Acquire cross-episode structure",True)
    p.line(181,394,181,470,arrow=True,dash=True)
    p.path([(36,522),(18,522),(18,245),(470,245),(470,219)],arrow=True,dash=True)
    p.path([(1064,352),(1080,352),(1080,245),(620,245),(620,219)],arrow=True,dash=True)
    p.text(400,490,"State writes ≠ fast-weight updates",21,weight=700)
    p.text(400,524,"Test memory, adaptation and replay separately.",17)
    p.text(36,598,"The implemented symbolic subset is shown in the separate prototype diagram.",16)
    p.save("roadmap.svg")

def results(e3):
    p=SVG("Synthetic development results", "Bars: means. Dots: five training seeds. Different benchmark versions; not a progress curve.",550)
    for panel,(title,scores) in enumerate([("E2 · no reliable event-specific advantage",e2_scores()),("E3 · positive oracle-grouping effect",e3["scores_by_condition"])]):
        offset=panel*550
        p.text(offset+36,115,title,18,weight=700)
        left=offset+196; width=270
        for tick in [0,25,50,75,100]:
            x=left+width*tick/100
            p.line(x,142,x,450)
            p.text(x,478,tick,14,anchor="middle")
        for i,key in enumerate(ORDER):
            y=160+i*58
            p.text(offset+36,y+21,LABELS[i],16)
            p.rect(left,y,mean(scores[key])*width,28,COLORS[i])
            for j,v in enumerate(scores[key]):
                p.circle(left+width*v,y+6+j*4,3.6,"#FFFFFF","#172C43")
            p.text(offset+480,y+21,f"{mean(scores[key])*100:.1f}",15,weight=700)
        p.text(left+width/2,510,"Original combined score (%)",16,anchor="middle")
    p.save("results.svg")
    p=SVG("E3 effect depends on the task", "Mean paired oracle-minus-fixed difference. Descriptive taskwise effects; no multiplicity adjustment.",625)
    tasks=list(e3["task_lag_scores_by_condition"]["oracle_events"])
    left=420; width=560
    def sx(value): return left+(value+15)/70*width
    for tick in [-10,0,10,20,30,40,50]:
        p.line(sx(tick),120,sx(tick),535,"#91A0AF" if tick==0 else "#E0E7ED",2 if tick==0 else 1)
        p.text(sx(tick),563,f"{tick:+d}" if tick else "0",15,anchor="middle")
    for i,key in enumerate(tasks):
        y=145+i*52
        difference=100*(mean(e3["task_lag_scores_by_condition"]["oracle_events"][key])-mean(e3["task_lag_scores_by_condition"]["fixed_chunks"][key]))
        label=key.replace("_"," ").replace("@"," · lag ")
        p.text(36,y+5,label,18)
        p.line(sx(0),y,sx(difference),y,"#007E78" if difference>=0 else "#B34F40",4)
        p.circle(sx(difference),y,6,"#007E78" if difference>=0 else "#B34F40")
        p.text(sx(difference)+(12 if difference>=0 else -12),y+5,f"{difference:+.2f}",16,weight=700,
               anchor="start" if difference>=0 else "end")
    p.text(700,605,"Difference (percentage points)",17,anchor="middle")
    p.save("task-effects.svg")
    p=SVG("E3 learning curves", "Five-seed mean on the same validation endpoint. The oracle curve is still rising at 1,000 updates.",630)
    left,top,width,height=85,140,925,345
    for tick in range(0,101,20):
        y=top+height*(1-tick/100);p.line(left,y,left+width,y);p.text(left-15,y+5,tick,15,anchor="end")
    for tick in [100,300,500,700,1000]:
        x=left+(tick-100)/900*width;p.text(x,513,tick,15,anchor="middle")
    for i,key in enumerate(ORDER):
        points=[(left+(v["step"]-100)/900*width,top+height*(1-v["score"])) for v in e3["mean_learning_curves"][key]]
        p.path(points,COLORS[i],3)
        for x,y in points:p.circle(x,y,3.5,COLORS[i])
        x=75+i*205;p.line(x,577,x+25,577,COLORS[i],4);p.text(x+33,582,LABELS[i],15)
    p.text(36,118,"Accuracy (%)",16);p.text(550,545,"Training updates",17,anchor="middle")
    p.save("learning-curves.svg")

def sherlock():
    data=load("results/cluster/1764075/result.json")
    human=next(v for v in data["summaries"] if v["condition"]=="human_events")
    random=data["random_matched_mean"]
    metrics=[("Primary · hit@5","subject_macro_best_iou_hit_at_5"),("Secondary · hit@1","subject_macro_best_iou_hit_at_1"),
             ("Secondary · reciprocal rank","mean_best_aligned_reciprocal_rank"),("Secondary · overlap nDCG@5","subject_macro_iou_ndcg_at_5")]
    p=SVG("Sherlock behavioral retrieval", "Human scene grouping does not improve the primary metric. Secondary outcomes remain exploratory.",595)
    for i,(label,key) in enumerate(metrics):
        x=36+(i%2)*550;y=125+(i//2)*205
        p.text(x,y,label,20,weight=700)
        for row,(name,value,color) in enumerate([("Human scenes",human[key],"#007E78"),("Random matched",random[key],"#A76B23")]):
            yy=y+28+row*47
            p.text(x,yy+21,name,16);p.rect(x+155,yy,value*290,29,color)
            p.text(x+155+value*290+8,yy+21,f"{value:.3f}",16)
        p.line(x+155,y+131,x+445,y+131)
        p.text(x+155,y+154,"0",14);p.text(x+445,y+154,"1",14,anchor="end")
    p.text(36,568,"401 development queries · 12 participants · BM25 costs not byte-matched · no raw fMRI",16)
    p.save("sherlock.svg")

def scoring_check():
    data=load("results/analysis/saved_endpoint_sensitivity_2026-11-11/sensitivity.json")
    p=SVG("The result depends on what is scored", "E3 only. Later scoring check; the original main measure has not been replaced.",560)
    panels=[("Ordinary answers", "answer_tasks_only"), ("Select a source record", "pointer_task_only")]
    for panel,(title,group_name) in enumerate(panels):
        offset=panel*550
        group=data["E3"]["groups"][group_name]
        p.text(offset+36,120,title,21,weight=700)
        left=offset+195; width=270
        for tick in [0,25,50,75,100]:
            x=left+width*tick/100
            p.line(x,148,x,365)
            p.text(x,391,tick,14,anchor="middle")
        for i,key in enumerate(ORDER[2:]):
            y=170+i*65
            p.text(offset+36,y+21,LABELS[i+2],16)
            score=group["means"][key]
            p.rect(left,y,score*width,30,COLORS[i+2])
            for j,value in enumerate(group["scores_in_seed_order"][key]):
                p.circle(left+width*value,y+6+j*4,3.6,"#FFFFFF","#172C43")
            p.text(offset+480,y+21,f"{score*100:.2f}",15,weight=700)
        p.text(left+width/2,425,"Accuracy (%)",16,anchor="middle")
        contrast=group["oracle_minus_fixed"]
        p.text(offset+36,465,f"Events minus fixed: {contrast['mean_delta']*100:+.2f} points",18,weight=700)
    p.text(36,514,"Dots show five training seeds on reused development questions. Different tasks, different target spaces.",16)
    p.save("scoring-check.svg")

def main():
    FIG.mkdir(parents=True,exist_ok=True)
    architecture()
    results(load("results/cluster/1806501_e3_final/summary.json"))
    sherlock()
    scoring_check()
    print("Generated 9 SVG figures from stored results and labeled architecture specifications.")

if __name__=="__main__":main()
