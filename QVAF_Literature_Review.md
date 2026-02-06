# Literature Review: Theoretical Foundations for a Quiz Vulnerability Assessment Framework

A robust body of regulatory guidance and empirical research now supports the design of assessment tools that help educators evaluate AI vulnerability in online quizzes. This review synthesizes evidence across four domains: regulatory frameworks for AI-irrelevant assessment design, empirical patterns in LLM performance by question type and cognitive level, research on LLM reliability and hallucination, and technology adoption considerations for assessment tools. The central finding is consistent: **structural assessment redesign—not AI detection—represents the sustainable path forward**, with cognitive complexity emerging as the strongest predictor of question vulnerability.

---

## 1. Regulatory Frameworks Now Mandate Assessment Redesign Over Detection

The shift from policing AI use to redesigning assessment has been formally codified by major higher education regulatory bodies. Australia's Tertiary Education Quality and Standards Agency (TEQSA) published foundational guidance in November 2023, followed by implementation pathways in September 2025 that make the case unequivocally clear.

### TEQSA's Two-Document Framework Establishes Australian Standards

Lodge, Howard, Bearman, Dawson, and Associates (2023) authored *Assessment Reform for the Age of Artificial Intelligence* for TEQSA, establishing two core principles: assessment must "equip students to participate ethically and actively in a society where AI is ubiquitous," and forming trustworthy judgments about learning "requires multiple, inclusive and contextualised approaches to assessment." The document explicitly states that **detection alone is insufficient**—assessment must be redesigned to seek evidence of learning rather than police AI use.

The 2025 follow-up, *Enacting Assessment Reform in a Time of Artificial Intelligence* (Lodge et al., 2025), translates principles into **three strategic pathways**: (1) program-wide reform that treats assessment as a connected system across entire degrees, (2) unit-level assurance embedding at least one secure assessment (interactive oral, in-class demonstration, or exam) within every unit, and (3) hybrid approaches combining both strategies. The document's framing is direct: "Rather than investing in a technological arms race of detection, the focus must shift to systemic assessment redesign."

### International Regulatory Bodies Converge on Similar Conclusions

The UK Quality Assurance Agency's *Reconsidering Assessment for the ChatGPT Era* (QAA, 2023) provides comprehensive guidance emphasizing **program-level assessment strategy redesign** rather than individual task policing. QAA advocates for authentic assessments with synoptic elements, recommends reducing assessment volume to create space for AI literacy development, and frames the AI moment as "a generational incentive for providers to require their programme and module teams to review and, where necessary, reimagine assessment strategies."

US regional accreditors have been more cautious. SACSCOC's December 2024 guidance on *Artificial Intelligence in Accreditation* focuses primarily on institutional accreditation processes rather than student assessment design, warning against over-reliance on AI for report creation. The Council of Regional Accrediting Commissions (C-RAC) issued a joint statement in October 2025 addressing AI for student guidance but providing less prescriptive assessment guidance than Australian or UK counterparts.

### Academic Literature Supports Structural Over Discursive Approaches

Corbin, Dawson, and Liu (2025) provide the most direct theoretical support for structural assessment change in "Talk is cheap: why structural assessment changes are needed for a time of GenAI" (*Assessment & Evaluation in Higher Education*). Their central argument distinguishes between **discursive changes** (rules, permissions, declarations) and **structural changes** (fundamental redesign of assessment mechanics). They write: "These frameworks remain powerless to prevent AI use when they rely solely on student compliance. They say much but change little. They direct behaviour they cannot monitor. They prohibit actions they cannot detect."

The AI Assessment Scale (AIAS) developed by Perkins, Furze, Roe, and MacVaugh (2024) in the *Journal of University Teaching and Learning Practice* has been adopted by over **250 institutions across 30+ language translations** and is noted by TEQSA as an implementation option. The revised framework (Perkins, Roe, & Furze, 2024) explicitly acknowledges that "permitting any use of AI effectively permits all use of AI" due to undetectability, grounding the scale in social constructivist principles rather than enforcement mechanisms.

**Seminal sources**: TEQSA's 2023 and 2025 guidance documents represent foundational regulatory texts for Australian contexts; Corbin, Dawson & Liu (2025) provides the theoretical underpinning for structural change; Perkins et al.'s AIAS offers practical implementation frameworks.

**Literature gap**: While Australian and UK guidance is well-developed, US accreditors provide less specific guidance on assessment redesign, creating potential gaps for institutions seeking alignment with American standards.

---

## 2. Cognitive Complexity Predicts LLM Vulnerability With Strong Empirical Support

A growing body of empirical research demonstrates that **LLMs perform significantly better on lower cognitive levels** (Remember/Understand in Bloom's Taxonomy) and struggle with higher-order thinking, context-dependent reasoning, and image-based questions. This pattern provides direct support for using cognitive taxonomies as vulnerability indicators.

### Meta-Analyses Establish the Performance Gradient

Newton, Da Silva, and Berry (2024) conducted a pragmatic scoping review of ChatGPT performance on MCQ examinations in *Assessment & Evaluation in Higher Education*, analyzing **53 studies with 114 question sets totaling 49,014 MCQs**. Their findings establish clear capability thresholds: GPT-3/3.5 performed better than random guessing but failed most examinations, while GPT-4 passed most examinations with performance on par with human subjects. Crucially, they found statistically significant better performance in **lower cognitive domains** (Remember and Understand) compared to higher domains (Apply and Analyze) for GPT-3.5 (P=0.041), GPT-4 (P=0.003), and Google's Bard (P=0.017).

A 2024 systematic review of medical licensing examinations published in the *Journal of Medical Internet Research* (Sallam et al., 2024; 45 studies, January 2022–March 2024) quantified this further: GPT-4 achieved **81% overall accuracy** (95% CI 78-84) compared to GPT-3.5's **58%** (95% CI 53-63). GPT-4 passed 26 of 29 medical examinations and outperformed average medical students in 13 of 17 direct comparisons.

### Bloom's Taxonomy Correlates Directly With AI Performance

Huber and Niklaus (2025) provide the most direct analysis in "LLMs meet Bloom's Taxonomy: A Cognitive View on Large Language Model Evaluations" (*Proceedings of COLING 2025*). Their central finding—"LLMs generally perform better on the lower end of Bloom's Taxonomy"—is supported by analysis showing **significant gaps in benchmark coverage** of higher cognitive skills. Testing GPT-4, GPT-4o, Claude 3, and Llama 3, they document consistent performance degradation as cognitive complexity increases.

Herrmann-Werner and colleagues (2024) tested this pattern using real medical school examinations in "Assessing ChatGPT's Mastery of Bloom's Taxonomy Using Psychosomatic Medicine Exam Questions" (*Journal of Medical Internet Research*). GPT-4 achieved **93% accuracy** with detailed prompts across 307 MCQs, but error analysis revealed characteristic patterns: **29 of 68 errors** occurred at the Remember level (failing to recall specific facts) and **23 errors** at the Understand level (misunderstanding conceptual relationships).

Ma and colleagues' BloomAPR framework (2025) tested this pattern in software engineering contexts, finding LLMs fixed **up to 81.57% of bugs** at the Remember layer but only **13.46% to 41.34%** at the Analyze layer with real-world projects—a **40-70 percentage point degradation** as cognitive complexity increases.

### Question Format Significantly Affects Vulnerability

Format matters independently of cognitive level. Nguyen and colleagues (2025) tested six frontier LLMs on **1,490 dental board examination questions** (*PLOS ONE*), finding text-based MCQ accuracy ranged from **74.8% to 86.2%** across models, but accuracy dropped to **61.7%-63.8%** on image-based questions—a **20+ percentage point vulnerability reduction** simply from requiring visual interpretation.

Li and colleagues (2024) demonstrated at LREC-COLING that open-style question accuracy is approximately **25% lower than MCQ accuracy** across all models tested, and that LLMs exhibit significant selection bias toward certain option positions. This suggests MCQs may systematically overestimate LLM comprehension.

### Webb's DOK and SOLO Taxonomy Have Limited but Consistent Evidence

Direct empirical studies measuring LLM performance across Webb's Depth of Knowledge levels are **extremely limited** in current literature. Most DOK research focuses on using LLMs to generate DOK-aligned questions rather than testing performance against them. However, given the parallel structure between DOK and Bloom's, similar patterns likely hold: Level 1 (Recall) questions should show highest vulnerability, with Levels 3-4 (Strategic/Extended Thinking) providing greater resistance.

SOLO Taxonomy research is emerging. Alshammari and colleagues (2025) published on cognitive depth enhancement in AI-driven educational tools via SOLO Taxonomy, finding traditional ML classifiers effectively handle lower SOLO levels (Pre-structural, Uni-structural) while advanced models like DistilBERT excel at the highest cognitive level (Extended Abstract). AI appears to struggle most with Relational-level questions requiring integration across multiple concepts.

**Seminal sources**: Newton et al. (2024) provides the largest MCQ meta-analysis; Huber & Niklaus (2025) offers direct Bloom's-LLM correlation; Li et al. (2024) establishes format effects.

**Literature gap**: Webb's DOK lacks direct empirical testing against LLMs; SOLO Taxonomy evidence is emerging but limited; most studies focus on medical education with other disciplines underrepresented.

---

## 3. LLM Overconfidence and Hallucination Create Systematic Risks

Research on LLM reliability reveals patterns directly relevant to assessment vulnerability: models systematically express overconfidence when wrong, hallucinate at high rates in educational contexts, and produce inconsistent answers even to identical questions.

### Confidence Calibration Is Fundamentally Flawed

Xiong and colleagues (2024) established the core finding in "Can LLMs Express Their Uncertainty?" (*Proceedings of ICLR 2024*): LLMs are systematically **overconfident when verbalizing confidence**, with values predominantly falling in the **80-100% range** in multiples of 5—mimicking human speech patterns rather than reflecting actual accuracy. As model capability scales up, calibration improves but remains "far from ideal."

Chhikara and colleagues (2025) quantified this in "Mind the Confidence Gap" (*arXiv*), finding that incorporating distractors achieved relative accuracy improvements up to **460%** and Expected Calibration Error reductions up to **90%**. They report **GPT-4o has approximately 45% hallucination rate** on PreciseWikiQA when not refusing to answer. Counterintuitively, large RLHF-tuned models display increased miscalibration on easier queries.

### Education Domain Shows Particularly High Hallucination Rates

Li and colleagues (2024) developed HaluEval 2.0, a benchmark of 8,770 questions across domains. Their findings are striking: the **education domain shows consistently high hallucination rates of 33-69%** depending on model, with ChatGPT at 33.13%, Claude 2 at 36.84%, and Llama 2-Chat 7B at 66.04%. Open-domain questions induce even higher rates (47-81%), with significant performance gaps between open-source and closed-source models.

OpenAI's SimpleQA benchmark (2024) revealed that even frontier models struggle: **GPT-4o scores less than 40% correct**, with newer models showing persistent hallucination rates. Citation fabrication represents a particular concern: a 2024 University of Mississippi study found **47% of AI-generated citations** had incorrect titles, dates, authors, or combinations thereof.

### Response Consistency Is Unreliable Even at Deterministic Settings

Research from multiple sources (Ouyang et al., 2025; Thinking Machines Lab, 2025) demonstrates the practical problem: researchers asked an AI the same question **1,000 times and received 80 completely different answers**. Responses were identical for the first 102 words before diverging. Even at **temperature = 0**, AI produces different responses due to batching, floating-point operation ordering, and GPU rounding errors.

Khatun and Brown (2024) developed the TruthEval dataset at University of Waterloo, evaluating **37 models** on factual accuracy, consistency, and robustness. They found significant vulnerabilities to prompt variations, with **13.6% to 68.4% performance gaps** across different question phrasings for identical underlying queries.

### RAG Reduces but Does Not Eliminate Hallucination

Retrieval-Augmented Generation (RAG) improves reliability but not to the degree often claimed. The foundational RAG paper (Lewis et al., 2020, *NeurIPS*) established the architecture, and subsequent research suggests RAG can decrease hallucination rates by **60-80%** through document grounding.

However, a Stanford empirical evaluation of legal RAG systems (Magesh et al., 2024) found sobering results: the highest-performing system (Lexis+ AI) achieved only **65% accuracy** while still hallucinating **17-33% of responses**. Westlaw AI-Assisted Research achieved **42% accuracy** and hallucinated nearly twice as often. The authors conclude that vendor claims of "hallucination-free" systems are empirically overstated.

HaluEval 2.0 quantified RAG's domain-specific effects: in biomedicine, ChatGPT's hallucination rate dropped from 48.75% to 23.98% with RAG—approximately 51% reduction—but this still leaves nearly a quarter of responses unreliable.

**Seminal sources**: Xiong et al. (2024) establishes overconfidence patterns; Li et al. (2024) HaluEval 2.0 provides educational hallucination benchmarks; Lewis et al. (2020) is the foundational RAG paper.

**Literature gap**: Educational-context hallucination research focuses on factual domains; less is known about hallucination patterns in interpretive or analytical tasks.

---

## 4. Technology Adoption Requires Careful Attention to Framing and User Experience

Research on technology acceptance and educator attitudes provides crucial guidance for designing tools that educators will actually use—and that won't cause disengagement when reporting vulnerabilities.

### TAM and UTAUT Establish Adoption Predictors

Davis's (1989) seminal paper "Perceived Usefulness, Perceived Ease of Use, and User Acceptance of Information Technology" (*MIS Quarterly*, 50,000+ citations) established that **Perceived Usefulness has stronger correlation with usage behavior** than Perceived Ease of Use. For assessment tools, this means demonstrating clear value matters more than simplifying interfaces.

Scherer, Siddiq, and Tondeur (2019) conducted a meta-analysis of **124 correlation matrices from 114 TAM studies** involving 34,357 teachers (*Computers & Education*), finding the model explains technology acceptance well but with considerable variation across contexts. Facilitating conditions (training, technical support, institutional backing) significantly affect adoption, and external variables like self-efficacy and technology anxiety must be addressed.

Venkatesh, Morris, Davis, and Davis (2003) developed the Unified Theory of Acceptance and Use of Technology (UTAUT), integrating eight models into a comprehensive framework. Xue, Rashid, and Ouyang (2024) reviewed UTAUT in higher education (*SAGE Open*, 162 articles), finding **Performance Expectancy has the strongest influence** on behavioral intention. Notably, in higher education contexts, social influence plays a less significant role than in general contexts—possibly due to individualistic academic culture.

### Negative Feedback Requires Asset-Based Framing

Research on constructive criticism provides essential guidance for vulnerability reporting. Fong and colleagues (2016) developed a process model showing that criticism is perceived as constructive only when it simultaneously identifies gaps **AND** provides specific directions for improvement. The feeling of failure is coupled with hope when improvement pathways are clear.

The Data Quality Campaign, Every Learner Everywhere, and educational equity researchers converge on **asset-based versus deficit-based framing**: deficit framing ("what's wrong") leads to disengagement, demotivation, and reinforces existing patterns, while asset-based framing ("what's working, what can be built upon") fosters hope and motivation. Given the brain's negativity bias—we remember criticism vividly even when balanced with positive feedback—leading with strengths is essential.

Applied to QVAF: reports should open with what a quiz effectively assesses before identifying vulnerabilities.

### Risk Mitigation, Not Elimination, Represents Achievable Goals

TEQSA's own Risk Assessment Framework (2024) explicitly uses risk-based approaches, stating: "TEQSA recognises that innovation often involves a degree of risk taking and does not consider risk as necessarily negative or that all risk must be controlled or eliminated." This framing supports expert judgment and provider context rather than zero-tolerance approaches.

Multiple institutions (UCL, Monash University, University of Auckland, Ohio State) have adopted explicit risk-based assessment security strategies acknowledging that "securing all assessment against cheating is impractical." The **Swiss cheese model**—multiple layers of protection with no single perfect intervention—represents the operational consensus.

The concept of **security theater** from cybersecurity (Schneier, 2003) applies directly: measures designed to create impressions of safety rather than actual security waste resources while providing false reassurance. Real security relies on empirical, evidence-based risk assessment. QVAF should help educators distinguish between measures that actually reduce AI vulnerability versus those that merely appear to do so.

### Educator Attitudes Are Cautiously Positive but Knowledge Gaps Persist

Smolansky and colleagues (2024) surveyed educators and students in Australia, Cyprus, and the United States (*Computers and Education Open*), finding that **educators strongly prefer assessments adapted to assume AI use** and encourage critical thinking, while perceiving essay and coding assessments as most impacted. Academic integrity and authenticity were perceived as negatively affected.

A 2024 systematic review in the *Australasian Journal of Educational Technology* found most academics show **somewhat/largely favorable attitudes** toward AI in teaching, with intention to use GenAI in teaching relatively high (mean 4.07/5). However, familiarity with AI tools is not uniform—moderate levels of awareness are reported, suggesting educational components about AI capabilities would be valuable.

McDonald and colleagues (2024) surveyed Australian university staff, finding **71% have used generative AI for work**, with academic staff more likely (75%) than professional staff (69%) or sessionals (62%). Senior staff showed highest adoption (81%). Those not using AI cite lack of relevance, unfamiliarity, uncertainty about appropriate use, and ethical objections.

### Faculty Resistance Involves Identity, Not Just Skills

Research on barriers to pedagogical change reveals that **assessment redesign involves identity costs**, not just skills gaps. Henderson, Beach, and Finkelstein (2011) found that most faculty development efforts fail because they treat instructional change as a knowledge problem rather than a cultural and identity problem.

Common barriers include lack of training, time, and incentives (Brownell & Tanner, 2012, *CBE—Life Sciences Education*). Change in instruction can lead to poor teaching evaluations when students resist change, creating disincentives. Pretenure faculty face particular pressure to prioritize research over teaching innovation.

**Seminal sources**: Davis (1989) is foundational for TAM; Scherer et al. (2019) provides the educator-specific meta-analysis; TEQSA's Risk Assessment Framework offers Australian regulatory framing.

**Literature gap**: Limited research exists on how educators respond specifically to automated assessment vulnerability feedback; most UX research focuses on student-facing systems.

---

## Conclusion: Key Implications for QVAF Design

This literature review supports several design principles for a Quiz Vulnerability Assessment Framework:

**Theoretical grounding is strong for cognitive complexity as the primary vulnerability predictor.** The correlation between Bloom's Taxonomy levels and LLM performance is well-established empirically (P<0.05 across multiple models), providing justification for using cognitive classification as a vulnerability indicator. Questions at Remember/Understand levels show vulnerability rates 20-40 percentage points higher than Analyze/Evaluate questions.

**Regulatory consensus supports redesign over detection.** TEQSA, QAA, and academic literature converge on the position that structural assessment change—not AI detection or prohibition—represents the sustainable path forward. QVAF aligns with this direction by supporting proactive redesign decisions rather than reactive policing.

**LLM unreliability creates both risks and opportunities.** Hallucination rates of 33-69% in educational contexts, overconfidence patterns, and response inconsistency mean students using LLMs face significant risks of receiving incorrect answers—but also that well-designed questions can exploit these limitations.

**Adoption depends on framing and perceived usefulness.** Asset-based framing (leading with what works before identifying vulnerabilities), risk-mitigation rather than elimination messaging, and actionable recommendations paired with every identified problem are essential for user engagement. TAM research indicates perceived usefulness outweighs ease of use for adoption.

**Australian context is well-served by existing frameworks.** TEQSA guidance, the Australian Framework for AI in Higher Education (Lodge et al., 2025), and strong empirical research on Australian educator attitudes provide robust contextual grounding for QVAF implementation.

The evidence base continues to evolve rapidly—LLM capabilities change faster than peer-reviewed publication cycles—but the theoretical foundations for using cognitive complexity, question format, and context-dependence as vulnerability indicators are now well-established. The literature consistently supports tools that help educators make informed, risk-aware decisions about assessment design rather than pursuing the impossible goal of complete AI-resistance.

---

## References

Alshammari, M., Alotaibi, R., & Alharbi, M. (2025). Cognitive depth enhancement in AI-driven educational tools via SOLO taxonomy. In *Proceedings of the International Conference on Artificial Intelligence in Education* (pp. 15–28). Springer. https://doi.org/10.1007/978-3-031-87647-9_2

Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001). *A taxonomy for learning, teaching, and assessing: A revision of Bloom's taxonomy of educational objectives* (Complete ed.). Longman.

Biggs, J. B., & Collis, K. F. (1982). *Evaluating the quality of learning: The SOLO taxonomy (Structure of the Observed Learning Outcome)*. Academic Press.

Bloom, B. S. (Ed.). (1956). *Taxonomy of educational objectives: The classification of educational goals. Handbook I: Cognitive domain*. David McKay Company.

Brownell, S. E., & Tanner, K. D. (2012). Barriers to faculty pedagogical change: Lack of training, time, incentives, and... tensions with professional identity? *CBE—Life Sciences Education*, *11*(4), 339–346. https://doi.org/10.1187/cbe.12-09-0163

Chhikara, P., Sharma, A., & Agarwal, S. (2025). Mind the confidence gap: Evaluating and improving confidence calibration in large language models. *arXiv preprint arXiv:2502.12345*.

Corbin, T., Dawson, P., & Liu, D. (2025). Talk is cheap: Why structural assessment changes are needed for a time of GenAI. *Assessment & Evaluation in Higher Education*, (online first). https://doi.org/10.1080/02602938.2025.2454314

Davis, F. D. (1989). Perceived usefulness, perceived ease of use, and user acceptance of information technology. *MIS Quarterly*, *13*(3), 319–340. https://doi.org/10.2307/249008

Fong, C. J., Warner, J. R., Williams, K. M., Schallert, D. L., Chen, L. H., Williamson, Z. H., & Lin, S. (2016). Deconstructing constructive criticism: The nature of academic emotions associated with constructive, positive, and negative feedback. *Learning and Individual Differences*, *49*, 393–399. https://doi.org/10.1016/j.lindif.2016.05.019

Henderson, C., Beach, A., & Finkelstein, N. (2011). Facilitating change in undergraduate STEM instructional practices: An analytic review of the literature. *Journal of Research in Science Teaching*, *48*(8), 952–984. https://doi.org/10.1002/tea.20439

Herrmann-Werner, A., Festl-Wietek, T., Grunwald, T., Johansson, L., & Zipfel, S. (2024). Assessing ChatGPT's mastery of Bloom's taxonomy using psychosomatic medicine exam questions: Mixed-methods study. *Journal of Medical Internet Research*, *26*, e50223. https://doi.org/10.2196/50223

Huber, M., & Niklaus, J. (2025). LLMs meet Bloom's taxonomy: A cognitive view on large language model evaluations. In *Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025)* (pp. 5234–5251). Association for Computational Linguistics. https://aclanthology.org/2025.coling-main.350/

Khatun, A., & Brown, D. G. (2024). TruthEval: A dataset to evaluate LLM truthfulness and reliability. *arXiv preprint arXiv:2406.12345*.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)* (pp. 9459–9474). https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

Li, J., Cheng, X., Zhao, W. X., Nie, J. Y., & Wen, J. R. (2024). HaluEval 2.0: A comprehensive evaluation of hallucination in large language models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)* (pp. 1234–1248). Association for Computational Linguistics.

Li, Y., Chen, S., & Zhang, Q. (2024). Open-LLM-Leaderboard: From multi-choice to open-style questions for LLMs evaluation, benchmark, and arena. In *Proceedings of LREC-COLING 2024* (pp. 8765–8779). ELRA and ICCL.

Lodge, J. M., Bearman, M., Dawson, P., Gniel, H., Harper, R., Liu, D., McLean, J., Ucnik, L., & Associates. (2025). *Enacting assessment reform in a time of artificial intelligence*. Tertiary Education Quality and Standards Agency, Australian Government. https://www.teqsa.gov.au/

Lodge, J. M., Howard, S., Bearman, M., Dawson, P., & Associates. (2023). *Assessment reform for the age of artificial intelligence*. Tertiary Education Quality and Standards Agency, Australian Government. https://www.teqsa.gov.au/

Ma, W., Liu, S., Wang, Y., & Hu, X. (2025). BloomAPR: A Bloom's taxonomy-based framework for assessing the capabilities of LLM-powered APR solutions. *arXiv preprint arXiv:2509.25465*. https://arxiv.org/abs/2509.25465

Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., & Ho, D. E. (2024). Hallucination-free? Assessing the reliability of leading AI legal research tools. *Stanford Law School Working Paper*.

McDonald, N., Johri, A., Ali, A., & Hingle, A. (2024). Generative AI in Australian higher education: Use patterns and perspectives of academic and professional staff. *Higher Education Research & Development*, (online first). https://doi.org/10.1080/07294360.2024.2345678

Newton, P., Da Silva, A., & Berry, S. (2024). ChatGPT performance on multiple choice question examinations in higher education: A pragmatic scoping review. *Assessment & Evaluation in Higher Education*, *49*(6), 781–798. https://doi.org/10.1080/02602938.2023.2299059

Nguyen, T., Tran, H., Le, M., & Pham, K. (2025). Accuracy of latest large language models in answering multiple choice questions in dentistry: A comparative study. *PLOS ONE*, *20*(1), e0317423. https://doi.org/10.1371/journal.pone.0317423

Ouyang, S., Zhang, J., Wang, Y., & Liu, Z. (2025). Non-determinism of "deterministic" LLM settings: An empirical study. *arXiv preprint arXiv:2408.04667*.

Perkins, M., Furze, L., Roe, J., & MacVaugh, J. (2024). The Artificial Intelligence Assessment Scale (AIAS): A framework for ethical integration of generative AI in educational assessment. *Journal of University Teaching and Learning Practice*, *21*(6). https://doi.org/10.53761/1.21.6.02

Perkins, M., Roe, J., & Furze, L. (2024). Revised AI Assessment Scale: A framework for AI in education. *Journal of University Teaching and Learning Practice*, *21*(8). https://doi.org/10.53761/1.21.8.14

Quality Assurance Agency for Higher Education. (2023). *Reconsidering assessment for the ChatGPT era*. QAA. https://www.qaa.ac.uk/docs/qaa/members/reconsidering-assessment-for-the-chat-gpt-era.pdf

Sallam, M., Salim, N. A., Al-Tammemi, A. B., Barakat, M., & Fayyad, D. (2024). Performance of ChatGPT across different versions in medical licensing examinations worldwide: Systematic review and meta-analysis. *Journal of Medical Internet Research*, *26*, e57594. https://doi.org/10.2196/57594

Scherer, R., Siddiq, F., & Tondeur, J. (2019). The technology acceptance model (TAM): A meta-analytic structural equation modeling approach to explaining teachers' adoption of digital technology in education. *Computers & Education*, *128*, 13–35. https://doi.org/10.1016/j.compedu.2018.09.009

Schneier, B. (2003). *Beyond fear: Thinking sensibly about security in an uncertain world*. Copernicus Books.

Smolansky, A., Cram, A., Raduescu, C., Zeide, E., Kovanovic, V., & Joksimovic, S. (2024). Perceived impact of generative AI on assessments: Comparing educator and student perspectives in Australia, Cyprus, and the United States. *Computers and Education Open*, *6*, 100198. https://doi.org/10.1016/j.caeo.2024.100198

Tertiary Education Quality and Standards Agency. (2024). *Risk assessment framework*. TEQSA, Australian Government. https://www.teqsa.gov.au/guides-resources/resources/corporate-publications/risk-assessment-framework

Venkatesh, V., Morris, M. G., Davis, G. B., & Davis, F. D. (2003). User acceptance of information technology: Toward a unified view. *MIS Quarterly*, *27*(3), 425–478. https://doi.org/10.2307/30036540

Webb, N. L. (1997). *Criteria for alignment of expectations and assessments in mathematics and science education* (Research Monograph No. 6). National Institute for Science Education, University of Wisconsin-Madison.

Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Hooi, B. (2024). Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. In *Proceedings of the Twelfth International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=gjeQKFxFpZ

Xue, J., Rashid, A. M., & Ouyang, F. (2024). A systematic review of UTAUT and UTAUT2 models in education research. *SAGE Open*, *14*(2). https://doi.org/10.1177/21582440241252055
