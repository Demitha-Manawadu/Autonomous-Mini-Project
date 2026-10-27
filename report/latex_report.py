"""Academic report content and genuine LaTeX PDF export."""
import re
import subprocess
import zipfile
from pathlib import Path
from PIL import Image

TITLE = 'Kalmann Filter Based Distance and Relative Speed Estimation for Autonomous Vehicle Safety'


def revise(pages, assets):
    p = lambda text: ('p', text)
    h = lambda text: ('h', text)
    table = lambda rows, caption: ('table', rows, caption)
    eq = lambda tex, plain: ('equation', tex, plain)
    intro = [h('Project background'),
        p('A vehicle needs to know how far an object is ahead and how quickly that gap is closing. These quantities support following-distance warnings and braking decisions. A noisy sensor can make the distance fluctuate. Estimating relative speed by subtracting consecutive readings makes the noise worse.'),
        p('We selected a linear Kalman filter to estimate distance and relative speed from range measurements. We tested it in two road scenarios: following a moving vehicle that brakes and approaching a stationary obstacle. A simple braking controller uses the estimated states to respond to closing traffic.'),
        h('Why this application was selected'),
        p('The application connects state estimation to a clear vehicle-safety problem. Both estimated quantities have a direct meaning: the gap tells how much space remains, and relative speed tells whether that space is shrinking. Moving and stationary targets also provide two different tests of the same estimator. Simulation makes it possible to compare each estimate with the exact state without requiring a vehicle or range-sensor hardware.'),
        h('Why a linear Kalman filter was selected'),
        p('For motion along one lane, the distance and relative-speed equations are linear. The range sensor measures distance directly, so the measurement model is linear too. A standard KF therefore fits the chosen model. An Extended KF would add linearisation without a need for it here. A Particle Filter would require a population of samples for a problem that can be represented by two state values and a covariance matrix.'),
        p('The KF combines a motion prediction with each new sensor reading. It can estimate relative speed even though the sensor measures only distance. It also reports uncertainty and can predict through short gaps in the observations. Its small computational cost makes it suitable for repeated updates.'),
        h('Objectives and scope'),
        p('The work aims to derive and implement the filter, demonstrate both road scenarios, measure errors against simulated truth, and examine tuning and sensor loss. The main question is whether filtering improves distance and relative-speed estimates compared with simple raw baselines.'),
        p('The model is longitudinal and tracks one known in-lane object. It does not perform mapping or steering. Lane membership is supplied by the simulator. Python is used for motion, measurements, filtering and plotting; the KF itself is implemented directly with matrix operations.')]
    revised=[('Introduction',intro)]
    setup=[]
    for block in pages[1][1]:
        if block[0]=='image':continue
        if block[0]=='h' and 'Aim and objectives' in block[1]:continue
        if block[0]=='p' and (block[1].startswith('A distance sensor') or block[1].startswith('The objectives')):continue
        setup.append(block)
    setup += [p('Both vehicles move in a fixed world coordinate system. In the following case, both rectangles advance along the road. In the obstacle case, only ego advances and the target stays fixed. The earlier ego-centred view was also valid, but a fixed road makes movement easier to see.'),
              p('Figure captions identify the scenario, sample time and seed. The numerical results use ground truth only for evaluation; the estimator and braking decision do not use the true target state.')]
    # Use an actual exported GIF frame on the page explaining the display.
    gif=assets.parents[1]/'outputs'/'following'/'simulation.gif'
    with Image.open(gif) as animation:
        animation.seek(min(100,animation.n_frames-1))
        animation.convert('RGB').save(assets/'following_gif_frame.png')
    setup.append(('image','following_gif_frame.png',
        'Frame from the exported following-scenario GIF at approximately 10 s. The upper view shows vehicle positions and sensor markers. The lower plot shows the true, measured and filtered gap, with a red cursor marking playback time.',15.5))
    revised.append(('Simulation arrangement',setup))
    revised.append(('Important parameters and scenario design',[
        p('Table 1 lists the main settings. The values define a manageable demonstration and are not claimed to be validated automotive limits. Both scenarios use the same filter and sensor settings.'),
        table([['Parameter','Value','Purpose'],
            ['Duration / sample interval','25 s / 0.1 s','251 samples; 10 updates per second'],
            ['Range noise standard deviation','2 m','Uncertainty in each range observation'],
            ['Process acceleration standard deviation','2 m/s²','Allows changes not captured by prediction'],
            ['Sensor forward range','180 m','Limits valid forward observations'],
            ['Initial relative-speed variance','100 (m/s)²','Represents unknown speed at start-up'],
            ['Initial ego speed','18 m/s','Same initial speed in both road cases'],
            ['Following initial gap','50 m','Both vehicles initially travel at 18 m/s'],
            ['Lead braking','−3 m/s² at 4–8 s','Lead reaches 6 m/s'],
            ['Stationary target position','100 m','Fixed in-lane obstacle; target speed is zero'],
            ['Ego braking command','−4 m/s²','Ideal deceleration; speed limited to zero'],
            ['Braking TTC threshold','5 s','Applied when closing faster than 0.3 m/s'],
            ['Stationary-target gap rule','Estimated gap < 12 m','Maintains braking near the fixed obstacle'],
            ['Noise seeds','42–61','20 repeated trials per road scenario']],
            'Main simulation and filter parameters.'),
        h('Reasons for the settings'),
        p('The 0.1 s interval provides a clear sequence of distance observations while keeping the animation easy to follow. The 25 s run includes motion before braking, the braking event and the response afterwards. A 2 m noise standard deviation makes measurement scatter visible, so the effect of filtering can be measured.'),
        p('Process uncertainty is set to 2 m/s² as a starting value and is varied later. It is a tuning parameter rather than a measured acceleration. The initial speed variance is large because one distance reading cannot reveal closing speed. The TTC and brake settings give a simple link between estimated motion and action, while keeping controller complexity outside the main filter study.'),
        p('The following case tests target braking. The stationary case tests a large initial closing speed and the transition to rest.')]))
    model=[]
    for b in pages[2][1]:
        if b[0]=='code' and b[1].startswith('x ='):
            model.extend([
                eq(r'\mathbf{x}_k=\begin{bmatrix}d_k\\v_{r,k}\end{bmatrix},\qquad v_{r,k}=v_{t,k}-v_{e,k}', 'x_k = [d_k, v_r,k]^T; v_r,k = v_target,k - v_ego,k'),
                eq(r'\begin{aligned}d_{k+1}&=d_k+\Delta t\,v_{r,k}+\tfrac12\Delta t^2 u_k,\\v_{r,k+1}&=v_{r,k}+\Delta t\,u_k.\end{aligned}', 'd_(k+1) = d_k + Δt v_r,k + ½Δt² u_k; v_r,(k+1) = v_r,k + Δt u_k')])
        elif b[0]=='code' and b[1].startswith('x(k+1)'):
            model.extend([
                eq(r'\mathbf{x}_{k+1}=\mathbf{F}\mathbf{x}_k+\mathbf{G}u_k+\mathbf{w}_k,\qquad z_k=\mathbf{H}\mathbf{x}_k+n_k', 'x_(k+1) = F x_k + G u_k + w_k; z_k = H x_k + n_k'),
                eq(r'\mathbf{F}=\begin{bmatrix}1&\Delta t\\0&1\end{bmatrix},\quad\mathbf{G}=\begin{bmatrix}\Delta t^2/2\\\Delta t\end{bmatrix},\quad\mathbf{H}=\begin{bmatrix}1&0\end{bmatrix},\quad u_k=-a_{e,k}', 'F = [[1, Δt], [0, 1]]; G = [Δt²/2, Δt]^T; H = [1, 0]; u_k = -a_ego,k')])
        elif b[0]=='code' and b[1].startswith('Q ='):
            model.extend([
                eq(r'\mathbf{Q}=\sigma_a^2\mathbf{G}\mathbf{G}^{\mathsf T},\quad R=\sigma_z^2,\quad \mathbf{w}_k\sim\mathcal{N}(\mathbf{0},\mathbf{Q}),\quad n_k\sim\mathcal{N}(0,R)', 'Q = σ_a² G G^T; R = σ_z²; w_k ~ N(0,Q); n_k ~ N(0,R)'),
                eq(r'\widehat{\mathbf{x}}_0=\begin{bmatrix}z_0\\0\end{bmatrix},\qquad\mathbf{P}_0=\begin{bmatrix}R&0\\0&100\end{bmatrix}', 'x̂_0 = [z_0, 0]^T; P_0 = diag(R, 100)')])
        elif b[0]=='p' and b[1].startswith('At dt ='):
            model.append(p('Using a 0.1 s sample interval, a process acceleration standard deviation of 2 m/s² and a range standard deviation of 2 m gives the following covariances.'))
            model.append(eq(r'\mathbf{Q}=\begin{bmatrix}0.0001&0.002\\0.002&0.04\end{bmatrix},\qquad R=4\ \mathrm{m^2}', 'Q = [[0.0001, 0.002], [0.002, 0.04]]; R = 4 m²'))
            model.append(p('The process model assumes an independent random relative acceleration held constant within each sample interval. Its tuning depends on the sampling interval.'))
        elif b[0]=='p' and b[1].startswith('Let d be'):
            model.append(p('The state contains the gap in metres and relative speed in metres per second. Relative speed is target speed minus ego speed. A negative value means the gap is closing.'))
        elif b[0]=='p':model.append(('p',b[1].replace(' Reference [1] describes the same class of linear tracking model.','')))
        else:model.append(b)
    revised.append(('Mathematical model',model))
    algorithm=[]
    for b in pages[3][1]:
        if b[0]=='code':
            if b[1].startswith('x_pred'):
                algorithm.append(eq(r'\widehat{\mathbf{x}}_k^- =\mathbf{F}\widehat{\mathbf{x}}_{k-1}+\mathbf{G}u_{k-1},\qquad\mathbf{P}_k^-=\mathbf{F}\mathbf{P}_{k-1}\mathbf{F}^{\mathsf T}+\mathbf{Q}', 'x̂_k^- = F x̂_(k-1) + G u_(k-1); P_k^- = F P_(k-1) F^T + Q'))
            elif b[1].startswith('innovation'):
                algorithm.extend([
                    eq(r'y_k=z_k-\mathbf{H}\widehat{\mathbf{x}}_k^-,\qquad S_k=\mathbf{H}\mathbf{P}_k^-\mathbf{H}^{\mathsf T}+R', 'y_k = z_k - H x̂_k^-; S_k = H P_k^- H^T + R'),
                    eq(r'\mathbf{K}_k=\mathbf{P}_k^-\mathbf{H}^{\mathsf T}S_k^{-1},\qquad\widehat{\mathbf{x}}_k=\widehat{\mathbf{x}}_k^-+\mathbf{K}_k y_k', 'K_k = P_k^- H^T / S_k; x̂_k = x̂_k^- + K_k y_k')])
            elif b[1].startswith('A ='):
                algorithm.append(eq(r'\mathbf{P}_k=(\mathbf{I}-\mathbf{K}_k\mathbf{H})\mathbf{P}_k^-(\mathbf{I}-\mathbf{K}_k\mathbf{H})^{\mathsf T}+\mathbf{K}_kR\mathbf{K}_k^{\mathsf T}', 'P_k = (I-K_k H) P_k^- (I-K_k H)^T + K_k R K_k^T'))
            else:
                algorithm.append(eq(r'\operatorname{TTC}=\begin{cases}0,&d\leq0,\\d/(-v_r),&d>0,\ v_r<-0.1\ \mathrm{m\,s^{-1}},\\\infty,&\text{otherwise}.\end{cases}', 'TTC = 0 if d ≤ 0; d/(-v_r) if d > 0 and v_r < -0.1 m/s; infinity otherwise'))
        else:algorithm.append(b)
    revised.append(('Kalman filter and braking decision',algorithm))
    revised.append(('Implementation',pages[4][1]))
    # Include an actual road screenshot as well as performance graphs for each case.
    for index,scenario,title in ((5,'following','Following a braking vehicle'),(6,'obstacle','Approaching a stationary obstacle')):
        im=Image.open(assets/f'{scenario}_road.png')
        im.crop((0,0,im.width,int(im.height*.46))).save(assets/f'{scenario}_scene.png')
        blocks=list(pages[index][1])
        blocks.insert(1,('image',f'{scenario}_scene.png',
            f"Road simulation for the {scenario} scenario, seed 42, at {'10' if scenario=='following' else '14'} s. The road is fixed in world coordinates; red ego indicates braking, while gray blocks are stationary roadside objects.",16))
        revised.append((title,blocks))
    evaluation=[]
    for b in pages[7][1]:
        if b[0]=='code':
            evaluation.extend([
                eq(r'\operatorname{RMSE}=\sqrt{\frac{1}{N}\sum_{k=1}^{N}\left(\widehat{x}_k-x_k^{\mathrm{true}}\right)^2}', 'RMSE = √[(1/N) Σ(x̂_k - x_true,k)²]'),
                eq(r'\widetilde{v}_{r,k}=\frac{z_k-z_{k-1}}{\Delta t},\qquad\sigma_{\Delta z/\Delta t}=\frac{\sqrt{2}\,\sigma_z}{\Delta t}', 'Raw relative speed = (z_k - z_(k-1))/Δt; difference noise std = √2 σ_z/Δt')])
        else:evaluation.append(b)
    revised.append(('Results and comparison',evaluation))
    revised.append(('Parameter tuning and sensor loss',pages[8][1]))
    revised.append(('Discussion and conclusion',pages[9][1]))
    reproduction=[]
    for b in pages[10][1]:
        if b[0]=='h' and 'References' in b[1]:break
        if b[0]=='p' and b[1].startswith('The report describes'):continue
        reproduction.append(b)
    reproduction.extend([
        h('Summary of the work'),
        p('We selected the linear Kalman filter because both the longitudinal motion model and the range measurement model are linear. We implemented its prediction and correction steps directly in Python, and used a simple road simulation to test the estimates before connecting them to a braking decision.'),
        p('We compared the estimated distance and relative speed with the known simulated states. The two road scenarios, repeated noise trials, tuning study and sensor-loss experiment show where the filter improves the estimates and where its assumptions limit performance. The equations, code excerpts, simulation figures and error tables document these findings.')])
    revised.append(('Running the project',reproduction))
    # Use the six main sections requested by the assignment, with descriptive
    # subsections rather than a long sequence of unrelated main headings.
    def grouped(items):
        blocks=[]
        for index,(title,part) in enumerate(items):
            if index:blocks.append(('break',))
            blocks.append(h(title))
            blocks.extend(('subh',b[1]) if b[0]=='h' else b for b in part)
        return blocks
    discussion=[]
    conclusion=[]
    in_conclusion=False
    for b in revised[10][1]:
        if b[0]=='h' and re.sub(r'^\d+(?:\.\d+)*\.?\s*','',b[1])=='Conclusion':
            in_conclusion=True
            continue
        (conclusion if in_conclusion else discussion).append(b)
    revised=[
        ('Introduction',revised[0][1]),
        ('Methodology',grouped(revised[1:5])),
        ('Implementation',revised[5][1]+[('break',)]+revised[11][1]),
        ('Results',grouped(revised[6:10])),
        ('Discussion',discussion),
        ('Conclusion',conclusion),
    ]
    figure_number=table_number=0
    for section,(_,blocks) in enumerate(revised,1):
        subsection=0
        subsubsection=0
        for index,b in enumerate(blocks):
            if b[0]=='h':
                subsection+=1
                subsubsection=0
                blocks[index]=('h',f'{section}.{subsection} '+re.sub(r'^\d+(?:\.\d+)*\.?\s*','',b[1]))
            elif b[0]=='subh':
                subsubsection+=1
                blocks[index]=('subh',f'{section}.{subsection}.{subsubsection} '+re.sub(r'^\d+(?:\.\d+)*\.?\s*','',b[1]))
            elif b[0]=='image':
                figure_number+=1
                caption=re.sub(r'^Figure\s+\d+\.\s*','',b[2])
                blocks[index]=('image',b[1],f'Figure {figure_number}. '+caption,b[3])
            elif b[0]=='table':
                table_number+=1
                caption=re.sub(r'^Table\s+\d+\.\s*','',b[2])
                blocks[index]=('table',b[1],f'Table {table_number}. '+caption)
    return revised


def latex_escape(t):
    changes={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_\allowbreak{}','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}',
             '²':r'\textsuperscript{2}','ᵀ':r'\textsuperscript{T}','−':'-','–':'--','—':'---','’':"'",'“':'``','”':"''",'×':r'\ensuremath{\times}','±':r'\ensuremath{\pm}','→':r'\ensuremath{\rightarrow}','Δ':r'\ensuremath{\Delta}','σ':r'\ensuremath{\sigma}','½':r'\ensuremath{\tfrac12}'}
    return ''.join(changes.get(c,c) for c in t)


def export_latex(pages, out, cover):
    esc=latex_escape
    lines=[r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=24mm,headheight=14pt]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,array,float}
\usepackage{caption,fancyhdr,eso-pic,needspace,listings,xcolor}
\usepackage[hidelinks,breaklinks]{hyperref}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\setlength{\parindent}{1.3em}
\setlength{\parskip}{4pt}
\setlength{\emergencystretch}{2em}
\linespread{1.08}
\setcounter{secnumdepth}{2}
\captionsetup{font=small,labelfont=bf,justification=centering}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small EN4594 -- Autonomous Systems}
\fancyhead[R]{}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\lstset{basicstyle=\ttfamily\footnotesize,breaklines=true,columns=fullflexible,keepspaces=true,frame=single,framerule=0.3pt,showstringspaces=false,aboveskip=8pt,belowskip=8pt}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\AddToShipoutPictureBG{\AtPageLowerLeft{\put(28.35,28.35){\setlength{\unitlength}{1pt}\linethickness{0.4pt}\framebox(538.57,785.19){}}}}
\begin{document}
\begin{titlepage}
\hypersetup{pageanchor=false}
\centering
\vspace*{6mm}
\includegraphics[height=43mm]{assets/uom_emblem.jpg}\par
\vspace{7mm}
{\Large University of Moratuwa\par}
\vspace{3mm}
{\Large Faculty of Engineering\par}
\vspace{3mm}
{\large Department of Electronic and Telecommunication Engineering\par}
\vspace{13mm}
{\large EN4594\par}
\vspace{2mm}
{\large Autonomous Systems\par}
\vspace{11mm}
{\huge\bfseries Mini Project Report\par}
\vspace{8mm}
{\LARGE\bfseries '''+esc(TITLE)+r'''\par}
\vspace{14mm}
{\Large '''+esc(cover['name'])+r'''\par}
\vspace{3mm}
{\large Index Number: '''+esc(cover['index'])+r'''\par}
\vspace{4mm}
{\large '''+esc(cover['date'])+r'''\par}
\vfill
{\large\bfseries Source Code Repository\par}
\vspace{3mm}
{\small\url{https://github.com/Demitha-Manawadu/Autonomous-Mini-Project.git}\par}
\vspace{10mm}
\end{titlepage}
\hypersetup{pageanchor=true}
\pagenumbering{roman}
\setcounter{tocdepth}{2}
\renewcommand{\contentsname}{Table of Contents}
\tableofcontents
\clearpage
\pagenumbering{arabic}
''']
    for title,blocks in pages:
        # Keep each main topic together, including its figures and discussion.
        lines.append(r'\clearpage\section{'+esc(title)+'}')
        for b in blocks:
            if b[0]=='p':
                text=b[1]
                if text.startswith('Repository: https://'):
                    lines.append(r'\noindent Source code: \url{'+text.split('Repository: ')[1]+r'}\par')
                else:
                    text=text.replace('report/report_evidence.json stores','The report_evidence.json file in the report folder stores')
                    lines.append(esc(text)+r'\par'+'\n')
            elif b[0] in ('h','subh'):
                heading=re.sub(r'^\d+(?:\.\d+)*\.?\s*','',b[1])
                level=r'\subsection' if b[0]=='h' else r'\subsubsection'
                lines.append(r'\needspace{4\baselineskip}'+level+'{'+esc(heading)+'}')
            elif b[0]=='break':lines.append(r'\clearpage')
            elif b[0]=='equation':lines.append(r'\begin{equation}'+b[1]+r'\end{equation}')
            elif b[0]=='code':lines.append('\\begin{lstlisting}\n'+b[1]+'\n\\end{lstlisting}')
            elif b[0]=='image':
                caption=re.sub(r'^Figure\s+\d+\.\s*','',b[2])
                width=min(b[3]/16.2,1.)
                lines.append(r'\begin{figure}[H]\centering\includegraphics[width='+f'{width:.3f}'+r'\linewidth]{assets/'+b[1]+r'}\caption{'+esc(caption)+r'}\end{figure}')
            elif b[0]=='table':
                rows=b[1];n=len(rows[0])
                caption=re.sub(r'^Table\s+\d+\.\s*','',b[2])
                lines.append(r'\begin{table}[H]\centering\caption{'+esc(caption)+r'}\small\renewcommand{\arraystretch}{1.2}\begin{tabularx}{\linewidth}{'+'Y'*n+r'}\toprule')
                for i,row in enumerate(rows):
                    cells=[esc(str(c)).replace('\n',' ') for c in row]
                    if i==0:cells=[r'\textbf{'+c+'}' for c in cells]
                    lines.append(' & '.join(cells)+r' \\'+('\n\\midrule' if i==0 else ''))
                lines.append(r'\bottomrule\end{tabularx}\end{table}')
    lines.append(r'\end{document}')
    source=out/'Manawadu_220381M_KF_Report.tex'
    source.write_text('\n'.join(lines),encoding='utf-8')
    engine=out/'tools'/'tectonic.exe'
    subprocess.run([str(engine),'--keep-logs',str(source.name)],cwd=out,check=True)
    with zipfile.ZipFile(out/'Manawadu_220381M_LaTeX_Source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(source,source.name)
        images=re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',source.read_text(encoding='utf-8'))
        for image in dict.fromkeys(images):archive.write(out/image,image)
        archive.writestr('BUILD.txt','Compile Manawadu_220381M_KF_Report.tex with pdfLaTeX or XeLaTeX. Keep assets beside the source. Run twice for the contents page and bookmarks.\n')
    print('Genuine LaTeX PDF created:',source.with_suffix('.pdf'))
