%%---------------------------------******-----------------------
%声明本文档为模板类
\ProvidesClass{homework-thesis}
    [2022/05/13 v1.0 update  https://github.com/mobtgzhang/latex/computervisionhomework]
%依赖的Latex版本
\NeedsTeXFormat{LaTeX2e}[2005/12/01]
%% Loading Class
% !TeX program = xelatex
\LoadClass[UTF8,a4paper,zihao=-4,space=auto]{ctexart}
\RequirePackage{setspace}
\onehalfspacing %1.5倍行距
%% 内容文字为小四，宋体，1.5倍行距。注意句首缩进和两端对齐。
%%---------------------------------******-----------------------
\RequirePackage{url}
% 设置字体
% 设置默认英文字体为Times New Roman.
\setmainfont{Times New Roman}
\setmonofont{Courier New} % 西文默认的等宽字体。
\setsansfont{Arial} % 西文默认无衬线字体
% 设置
\setCJKmainfont[AutoFakeBold = {2.15},ItalicFont={KaiTi}]{SimSun}
\setCJKfamilyfont{xw}{STXinwei}
%\setCJKfamilyfont{zhyou}{YouYuan}
\ifx\lishu\undefined%
 \setCJKfamilyfont{zhli}{LiSu}
 \newcommand*{\lishu}{\CJKfamily{zhli}} %% Do this if it is undefined
\else
  %% Do this if it is defined
\fi
\newcommand*{\xinwei}{\CJKfamily{xw}} 
%%---------------------------------******-----------------------
%% 引入toprule,midrule,bottomrule
\RequirePackage{booktabs}
%% 引入数学宏包
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{bm}
\RequirePackage{mathrsfs}
% 每行缩进两个汉字
\setlength\parindent{2em}
%% enumerate 列表环境间距调节
\RequirePackage{enumitem}

% 页面布局
\RequirePackage{geometry}
\geometry{
    top=1in,
    bottom=1in,
    left=1.25in,
    right=1.25in,
    headsep=8mm
}
% 表格
\RequirePackage{tabularx,array}
%% 插入图
\RequirePackage{graphicx}
%% 如果插入的图片没有指定扩展名，那么依次搜索下面的扩展名所对应的文件
\DeclareGraphicsExtensions{.pdf,.eps,.png,.jpg,.jpeg}
\graphicspath{{figures/}}

%% 并列子图
\RequirePackage{subfigure}
\RequirePackage{wrapfig}
% 英语加粗
\RequirePackage[english]{babel}
% 颜色
\RequirePackage{color}
%% 参考文献设置
\RequirePackage[numbers,square,comma,super,sort&compress]{natbib}
% 上标引用
\newcommand{\upcite}[1]{\textsuperscript{\cite{#1}}}
%% -------------------------------*************----------
%定义使用者需要填写的标签
\def\homework@label@facultyname{学\qquad 院~}
\def\homework@label@classnumber{班级学号~}
\def\homework@label@professional{专\qquad 业~}
\def\homework@label@studentname{学\qquad 生~}
\def\homework@label@researchdirection{研究方向~}
\def\homework@label@covertitle{题~目~}

%定义上述标签的默认值
\def\homework@value@facultyname{电子电气工程学院}
\def\homework@value@classnumber{M123456789}
\def\homework@value@studentname{张~三}
\def\homework@value@professional{专~业}
\def\homework@value@researchdirection{\textcolor{red}{如果未确定，就填待定}}
\def\homework@value@covertitlefirstline{}
\def\homework@value@covertitlesecondline{}

%定义用户填写上述标签对应值的命令,需要用户在主文档自行调用
\newcommand\facultyname[1]{\def\homework@value@facultyname{#1}}
\newcommand\classnumber[1]{\def\homework@value@classnumber{#1}}
\newcommand\studentname[1]{\def\homework@value@studentname{#1}}
\newcommand\professional[1]{\def\homework@value@professional{#1}}
\newcommand\researchdirection[1]{\def\homework@value@researchdirection{#1}}
\newcommand\covertitle[1]{\def\homework@value@covertitle{#1}}
\newcommand\covertitlefirstline[1]{\def\homework@value@covertitlefirstline{#1}}
\newcommand\covertitlesecondline[1]{\def\homework@value@covertitlesecondline{#1}}

%% 自定义论文的封面
\newcommand{\makecover}{
	\newpage\thispagestyle{empty}
	\vspace{34pt}
	\begin{center}
    \linespread{1.3}

		{{\heiti\zihao{1}\bfseries{上~海~工~程~技~术~大~学\\}}}

    {{\heiti\zihao{1}\bfseries{计算机视觉大作业报告\\}}}

    \vspace{20pt}

    {\zihao{5}\textcolor{red}{红字为说明内容，提交前请删除}}
  \end{center}
  \begin{center}
		\vspace{120pt}
		% Please add the following required packages to your document preamble:
		\songti\zihao{4}{
      \renewcommand{\arraystretch}{1.3}%调行距
			\begin{tabular}{p{1.10in}<{\centering}p{3.38in}<{\centering}}
      \homework@label@facultyname & \homework@value@facultyname\\
      \cmidrule{2-2}
			\homework@label@professional & \homework@value@professional\\
      \cmidrule{2-2}
      \homework@label@classnumber & \homework@value@classnumber\\
      \cmidrule{2-2}      
      \homework@label@studentname & \homework@value@studentname\\
      \cmidrule{2-2}      
      \homework@label@researchdirection & \homework@value@researchdirection\\
      \cmidrule{2-2}   
			\end{tabular}		
		}

		\vspace{42pt}
	\end{center}
  \begin{center}
    \songti\zihao{-2}{
			\begin{tabular}{p{1.03in}<{\centering}p{4.36in}<{\centering}}
      \homework@label@covertitle & \homework@value@covertitlefirstline\\
      \cmidrule{2-2}
			 & \homework@value@covertitlefirstline\\
      \cmidrule{2-2}
			\end{tabular}		
		}
  \end{center}
  \begin{flushleft}
    \songti\zihao{4}\textcolor{red}{（题目任选，但需要在数字图像处理技术范围内，包括但不限于图像增强、图像去噪、边缘检测、角点检测、纹理特征提取、SIFT、SURF等等，如果自己没有把握，担心不在范围内，请直接与我联系确认。） }{\footnote{\bfseries{本latex模板作者:mobtgzhang, 更新地址\\ \url{https://github.com/mobtgzhang/latex/computervisionhomework}}}}
  \end{flushleft}
  \newpage
}


\RequirePackage{fancyhdr}   % 设置页眉、页脚

%%定义正文的页眉页脚格式
\def\homework@section{}
\fancypagestyle{homework@heading}{
	\fancyhf{}
	\fancyhead[C]{}
	\fancyfoot[C]{\thepage}
}
%设定文档页眉页脚
\pagestyle{empty}


%%-----------------------------*****************----------------


%% 设置图表标题选项
\RequirePackage{caption}
\DeclareCaptionLabelSeparator{zhspace}{\hspace{1\ccwd}}
\DeclareCaptionFont{fontsize}{\zihao{5}}
\captionsetup{
	font = {fontsize},
	labelsep = zhspace,
}
\captionsetup[table]{
	position = top,
	aboveskip = 6bp,
	belowskip = 6bp,
}

\captionsetup[figure]{
	position = bottom,
	aboveskip = 6bp,
	belowskip = 6bp,
}
% 将浮动参数设为较宽松的值
\renewcommand{\textfraction}{0.15}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{0.65}
\renewcommand{\floatpagefraction}{0.60}

% 定公式、图、表编号为"3-1"的形式，即分隔符由.变为短杠
\renewcommand\theequation{\arabic{section}-\arabic{equation}}
\renewcommand\thefigure{\arabic{section}-\arabic{figure}}
\renewcommand\thetable{\arabic{section}-\arabic{table}}

% 中文破折号
\newcommand{\cndash}{\rule{0.0em}{0pt}\rule[0.35em]{1.4em}{0.05em}\rule{0.2em}{0pt}}

% listings 源代码显示宏包
\RequirePackage{listings}
\lstset{tabsize=4, %
	frame=shadowbox, %把代码用带有阴影的框圈起来
	commentstyle=\color{red!50!green!50!blue!50},%浅灰色的注释
	rulesepcolor=\color{red!20!green!20!blue!20},%代码块边框为淡青色
	keywordstyle=\color{blue!90}\bfseries, %代码关键字的颜色为蓝色，粗体
	showstringspaces=false,%不显示代码字符串中间的空格标记
	stringstyle=\ttfamily, % 代码字符串的特殊格式
	keepspaces=true, %
	breakindent=22pt, %
	numbers=left,%左侧显示行号
	stepnumber=1,%
	numberstyle=\tiny, %行号字体用小号
	basicstyle=\footnotesize, %
	showspaces=false, %
	flexiblecolumns=true, %
	breaklines=true, %对过长的代码自动换行
	breakautoindent=true,%
	breakindent=4em, %
	aboveskip=1em, %代码块边框
	fontadjust,
	captionpos=t,
	framextopmargin=2pt,framexbottommargin=2pt,abovecaptionskip=-3pt,belowcaptionskip=3pt,
	xleftmargin=4em,xrightmargin=4em, % 设定listing左右的空白
	texcl=true,
	% 设定中文冲突，断行，列模式，数学环境输入，listing数字的样式
	extendedchars=false,columns=flexible,mathescape=true
	numbersep=-1em
}


%_ 参考文献环境
\renewenvironment{thebibliography}[1]
{\zihao{5}
	\section{参考文献}
	\@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}%
	\addcontentsline{toc}{section}{参考文献}
	\list{\@biblabel{\@arabic\c@enumiv}}%
	{\settowidth\labelwidth{\@biblabel{#1}}%
		\leftmargin\labelwidth
		\advance\leftmargin\labelsep
		 \setlength{\baselineskip}{16pt}
		\setlength{\parsep}{-0.5mm}
		\setlength{\labelsep}{0.5em}
		%\setlength{\itemsep}{0.5pc}
		\setlength{\listparindent}{0in}
		\setlength{\itemindent}{0in}
		\setlength{\rightmargin}{0in}
		\@openbib@code
		\usecounter{enumiv}%
		\let\p@enumiv\@empty
		\renewcommand\theenumiv{\@arabic\c@enumiv}}%
	\sloppy
	\clubpenalty4000
	\@clubpenalty \clubpenalty
	\widowpenalty4000%
	\sfcode`\.\@m}
{\def\@noitemerr
	{\@latex@warning{Empty `thebibliography' environment}}%
	\endlist}


%% 设置章节格式
%% 正文当中一级标题采用宋体四号字，居左书写；
\RequirePackage{zhnumber} % change section number to chinese
\ctexset{section={
      format={\raggedright \bfseries\songti  \zihao{4}},
      beforeskip = 28bp plus 1ex minus .2ex,
      afterskip = 24bp plus .2ex,
      fixskip = true,
  }
}


% 设置二级标题格式
%% 正文当中二级标题采用宋体四号字，居左书写；
\ctexset{subsection={
    format = { \bfseries\songti \raggedright \zihao{4}},
    beforeskip =28bp plus 1ex minus .2ex,
    afterskip = 24bp plus .2ex,
    fixskip = true,
  }
}
% 设置三级标题格式
%% 正文当中三级标题采用宋体四号字，居左书写；
\ctexset{subsubsection={
    format = { \bfseries\songti \raggedright \zihao{4}},
    beforeskip =28bp plus 1ex minus .2ex,
    afterskip = 24bp plus .2ex,
    fixskip = true,
  }
}

\renewcommand\thesection{\zhnum{section}、}
\renewcommand\thesubsection{\arabic{section}.\arabic{subsection}}
\renewcommand\thesubsubsection{\arabic{section}.\arabic{subsection}.\arabic{subsubsection}}
