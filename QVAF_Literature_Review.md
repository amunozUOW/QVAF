# Literature Review: Theoretical Foundations for a Quiz Vulnerability Assessment Framework

## Introduction

This review synthesises regulatory guidance and empirical research relevant to the design of assessment tools that evaluate AI vulnerability in online quizzes. The review addresses five domains: regulatory frameworks for assessment design in AI contexts, empirical patterns in large language model (LLM) performance by question type and cognitive level, research on LLM reliability and hallucination, technology adoption considerations for assessment tools, and design rationale for the Quiz Vulnerability Assessment Framework (QVAF). The synthesis indicates that structural assessment redesign represents a more sustainable approach than AI detection, with cognitive complexity serving as a predictor of question vulnerability.

---

## 1. Regulatory Frameworks for Assessment Redesign

Regulatory bodies in Australia, the United Kingdom, and the United States have issued guidance on assessment practices in the context of generative AI. This section examines these frameworks and their implications for assessment design.

### 1.1 Australian Regulatory Guidance

The Tertiary Education Quality and Standards Agency (TEQSA) published two guidance documents that establish principles for assessment reform. Lodge et al. (2023) authored *Assessment Reform for the Age of Artificial Intelligence*, which articulated two principles: that assessment should prepare students for participation in contexts where AI is prevalent, and that trustworthy judgments about learning require multiple approaches to assessment. The document positioned detection as insufficient as a standalone strategy, advocating instead for assessment redesign.

Lodge et al. (2025) published *Enacting Assessment Reform in a Time of Artificial Intelligence*, which outlined three implementation pathways: program-wide reform treating assessment as a connected system across degree programs; unit-level assurance embedding at least one secure assessment within each unit; and hybrid approaches combining both strategies.

### 1.2 International Regulatory Guidance

The Quality Assurance Agency for Higher Education (QAA, 2023) published *Reconsidering Assessment for the ChatGPT Era*, which emphasised program-level assessment strategy redesign. The document advocated for authentic assessments with synoptic elements and recommended reducing assessment volume to create capacity for AI literacy development.

In the United States, the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC, 2024) issued guidance focusing on institutional accreditation processes rather than student assessment design. The Council of Regional Accrediting Commissions (C-RAC, 2025) issued a statement addressing AI applications for credit transfer and learning evaluation, though this provided less prescriptive assessment guidance than Australian or UK counterparts.

### 1.3 Academic Literature on Assessment Change

Corbin et al. (2025) distinguished between discursive changes (rules, permissions, declarations) and structural changes (redesign of assessment mechanics) in their analysis published in *Assessment & Evaluation in Higher Education*. The authors argued that frameworks relying on student compliance remain limited in their capacity to prevent AI use.

Perkins et al. (2024a) developed the AI Assessment Scale (AIAS), published in the *Journal of University Teaching and Learning Practice*, which has been adopted by institutions across multiple language translations and is referenced by TEQSA as an implementation option. The revised framework (Perkins et al., 2024b) acknowledged that permitting AI use in assessments creates challenges for enforcement due to detection limitations, grounding the scale in social constructivist principles.

---

## 2. Cognitive Complexity and LLM Performance

Empirical research indicates that LLMs perform differently across cognitive levels, with performance patterns varying by question type and format. This section reviews evidence on these performance patterns.

### 2.1 Performance on Examinations

Newton and Xiromeriti (2024) conducted a scoping review of ChatGPT performance on multiple-choice question (MCQ) examinations, analysing 53 studies comprising 114 question sets and 49,014 MCQs. The review found that GPT-3 and GPT-3.5 performed above chance but failed most examinations, while GPT-4 passed most examinations with performance comparable to human subjects.

Sallam et al. (2024) examined ChatGPT performance on radiology board-style examinations and found statistically significant differences in performance across cognitive domains, with higher performance on Remember and Understand levels compared to Apply and Analyze levels for GPT-3.5 (*p* = .041), GPT-4 (*p* = .003), and Google Bard (*p* = .017).

Liu et al. (2024) conducted a systematic review and meta-analysis of ChatGPT performance on medical licensing examinations (45 studies, January 2022–March 2024). GPT-4 achieved 81% accuracy (95% CI [78, 84]) compared to 58% for GPT-3.5 (95% CI [53, 63]). GPT-4 passed 26 of 29 medical examinations and outperformed average medical student performance in 13 of 17 comparisons.

### 2.2 Cognitive Taxonomy and Performance

Huber and Niklaus (2025) analysed LLM performance in relation to Bloom's Taxonomy in a study presented at COLING 2025. The authors found that LLMs performed better on lower levels of Bloom's Taxonomy and identified gaps in benchmark coverage of higher cognitive skills. Testing included GPT-4, GPT-4o, Claude 3, and Llama 3.

Herrmann-Werner et al. (2024) tested GPT-4 using medical school examination questions and Bloom's Taxonomy classification. GPT-4 achieved 92.5% accuracy with detailed prompts across 307 MCQs. Error analysis showed that 29 of 68 errors occurred at the Remember level and 23 errors at the Understand level.

Ma et al. (2025) developed the BloomAPR framework and tested LLM performance on software bug repair across cognitive levels. LLMs fixed up to 81.57% of bugs at the Remember level but 13.46% to 41.34% at the Analyze level in real-world projects.

### 2.3 Question Format Effects

Nguyen et al. (2025) tested six LLMs on 1,490 dental board examination questions. Text-based MCQ accuracy ranged from 74.8% to 86.2% across models, while accuracy on image-based questions ranged from 61.7% to 63.8%.

Myrzakhan et al. (2024) examined performance differences between MCQ and open-style questions in their Open-LLM-Leaderboard study. Open-style question accuracy was lower than MCQ accuracy across models tested, and LLMs exhibited selection bias toward certain option positions.

### 2.4 Alternative Cognitive Taxonomies

Research on LLM performance across Webb's Depth of Knowledge levels remains limited. Most existing DOK research has focused on using LLMs to generate DOK-aligned questions rather than testing performance against them.

Yaacoub et al. (2025) examined AI performance in relation to SOLO Taxonomy, finding that traditional machine learning classifiers performed adequately at lower SOLO levels (Pre-structural, Uni-structural) while transformer-based models such as DistilBERT showed stronger performance at the Extended Abstract level.

---

## 3. LLM Reliability and Hallucination

Research on LLM reliability identifies patterns relevant to assessment vulnerability, including overconfidence in incorrect responses, hallucination across domains, and response inconsistency.

### 3.1 Confidence Calibration

Xiong et al. (2024) examined confidence elicitation in LLMs in a study presented at ICLR 2024. The authors found that LLMs expressed overconfidence when verbalising confidence levels, with values predominantly in the 80–100% range in multiples of five. Calibration improved with model capability but remained imperfect.

Chhikara et al. (2025) examined confidence calibration in LLMs and found that incorporating distractors achieved relative accuracy improvements up to 460% and Expected Calibration Error reductions up to 90%. The study also found that RLHF-tuned models displayed increased miscalibration on easier queries.

### 3.2 Hallucination Rates

Li et al. (2024) developed a hallucination benchmark presented at ACL 2024, examining factuality across domains. The education domain exhibited high hallucination rates, with performance gaps between open-source and closed-source models. Open-domain questions induced higher hallucination rates than domain-specific questions.

OpenAI (2024) released the SimpleQA benchmark, which found that GPT-4o achieved approximately 38% accuracy on factual questions.

Walters and Wilanda (2023) examined citation fabrication in AI-generated reference lists, finding fabrication rates of 47–69% depending on prompt type and model version.

### 3.3 Response Consistency

Ouyang et al. (2025) examined non-determinism in LLM outputs and found that LLMs produce different responses even at temperature = 0 due to batching, floating-point operation ordering, and GPU parallelism effects. Thinking Machines Lab (2025) demonstrated this pattern by submitting 1,000 identical prompts and receiving 80 different responses, with divergence occurring after approximately 100 tokens.

Khatun and Brown (2024) developed the TruthEval dataset, evaluating 37 models on factual accuracy, consistency, and robustness. The study found that models showed vulnerability to prompt variations, with performance gaps across different phrasings of identical queries.

### 3.4 Retrieval-Augmented Generation

Lewis et al. (2020) introduced Retrieval-Augmented Generation (RAG) for knowledge-intensive NLP tasks at NeurIPS 2020, establishing an architecture for grounding LLM responses in retrieved documents.

Magesh et al. (2024) evaluated legal RAG systems and found that systems hallucinated 17–33% of responses. The authors concluded that vendor claims regarding hallucination-free systems were not supported by empirical evidence.

Research on domain-specific RAG effects indicates reductions in hallucination rates; in biomedical contexts, hallucination rates decreased from approximately 49% to 24% with RAG implementation.

---

## 4. Technology Adoption Considerations

Research on technology acceptance and educator attitudes provides guidance for designing assessment tools that educators will adopt.

### 4.1 Technology Acceptance Models

Davis (1989) established the Technology Acceptance Model (TAM), finding that Perceived Usefulness had stronger correlation with usage behaviour than Perceived Ease of Use. This paper has received over 60,000 citations.

Scherer et al. (2019) conducted a meta-analysis of 124 correlation matrices from 114 TAM studies involving 34,357 teachers. The study found that the model explained technology acceptance with variation across contexts, and that facilitating conditions (training, technical support, institutional backing) affected adoption.

Venkatesh et al. (2003) developed the Unified Theory of Acceptance and Use of Technology (UTAUT), integrating eight models. Xue et al. (2024) reviewed UTAUT in higher education contexts (162 articles) and found that Performance Expectancy had the strongest influence on behavioural intention. Social influence showed less significance in higher education contexts compared to general contexts.

### 4.2 Feedback Framing

Fong et al. (2018) developed a process model of constructive criticism and found that criticism is perceived as constructive when it identifies gaps and provides directions for improvement.

Research on asset-based versus deficit-based framing indicates that deficit framing leads to disengagement, while asset-based framing supports motivation (Data Quality Campaign; Every Learner Everywhere).

### 4.3 Risk-Based Approaches

TEQSA (2024) published a Risk Assessment Framework stating that "TEQSA recognises that innovation often involves a degree of risk taking and does not consider risk as necessarily negative or that all risk must be controlled or eliminated."

Schneier (2003) introduced the concept of security theatre, referring to measures designed to create impressions of safety rather than actual security.

### 4.4 Educator Attitudes

Kizilcec et al. (2024) surveyed educators and students in Australia, Cyprus, and the United States. Educators reported preferences for assessments adapted to assume AI use and perceived essay and coding assessments as most affected by AI.

McDonald et al. (2024) surveyed Australian university staff and found that 71% had used generative AI for work. Academic staff reported higher usage (75%) than professional staff (69%) or sessional staff (62%). Senior staff showed the highest adoption rate (81%).

### 4.5 Barriers to Pedagogical Change

Henderson et al. (2011) reviewed literature on change in undergraduate STEM instructional practices and found that faculty development efforts that treated instructional change as a knowledge problem rather than a cultural and identity problem were less effective.

Brownell and Tanner (2012) identified barriers to faculty pedagogical change including lack of training, time, and incentives. Instructional change can lead to lower teaching evaluations when students resist change.

---

## 5. QVAF Design Rationale and Methodology

The preceding literature provides foundations for QVAF design decisions. This section documents how these foundations translate into specific design choices.

### 5.1 Problem Scope

QVAF addresses a specific behaviour: students copying quiz questions into an AI system and using the AI-generated answer. This scope excludes:

- Contract cheating involving human assistance
- Sophisticated prompt engineering
- Real-time AI assistance during proctored examinations
- AI-assisted learning followed by legitimate recall
- Multi-modal cheating using image capture

### 5.2 Cognitive Demand Taxonomy

QVAF employs a five-level cognitive demand taxonomy synthesising Bloom's Revised Taxonomy (Anderson & Krathwohl, 2001), Webb's Depth of Knowledge (Webb, 1997), and SOLO Taxonomy (Biggs & Collis, 1982):

| Level | Name | Description |
|-------|------|-------------|
| 1 | Recall | Direct retrieval of memorised facts, definitions, or procedures |
| 2 | Routine Application | Applying known procedures where method selection is apparent |
| 3 | Conceptual Understanding | Demonstrating understanding of relationships between concepts |
| 4 | Analytical Reasoning | Breaking down information; evaluating evidence |
| 5 | Strategic Integration | Synthesising multiple sources; applying to novel situations |

### 5.3 Testing Methodology

QVAF tests each question under two conditions:

| Condition | Simulation | Information Provided |
|-----------|------------|---------------------|
| Baseline (no RAG) | Student copying question into general-purpose AI | Vulnerability to general AI knowledge |
| Enhanced (with RAG) | Student using AI with access to course materials | Vulnerability from course-specific content |

The combination of baseline and RAG results produces four patterns:

| Pattern | Baseline | RAG | Interpretation |
|---------|----------|-----|----------------|
| Correct Both | ✓ | ✓ | AI succeeds regardless of materials access |
| Correct RAG Only | ✗ | ✓ | Course materials enable AI success |
| Incorrect Both | ✗ | ✗ | Question resists AI assistance |
| Correct Baseline Only | ✓ | ✗ | RAG may introduce confusion |

### 5.4 Multi-Sample Testing Rationale

LLMs produce variable outputs even at temperature = 0 due to floating-point operation ordering, GPU parallelism effects, and batching variations (Ouyang et al., 2025). Multi-sample testing captures this variability: a question answered correctly 10/10 times differs from one answered correctly 7/10 times.

### 5.5 Design Philosophy

QVAF functions as a decision-support tool rather than a prescriptive system:

| Tool Function | Educator Function |
|---------------|-------------------|
| Provides metrics | Interprets metrics in context |
| Classifies cognitive demand | Validates or overrides classifications |
| Generates recommendations | Accepts, modifies, or rejects suggestions |
| Identifies patterns | Decides on action |

The framework avoids categorical vulnerability labels (e.g., "HIGH RISK") because:

1. Context determines interpretation (70% AI accuracy may be acceptable in formative quizzes but concerning in high-stakes examinations)
2. Stakes vary across assessments
3. Educators retain professional autonomy in assessment decisions
4. Disciplinary contexts differ

### 5.6 Limitations

QVAF is designed for unproctored online quizzes with unrestricted external resource access, MCQ, true/false, and short-answer formats, and single LLM testing using a representative model.

| Limitation | Implication | Mitigation |
|------------|-------------|------------|
| Single model testing | Results reflect one AI system | Use capable, representative model; acknowledge limitation |
| LLM-based classification | Cognitive demand classification uses the same technology being tested | Treat classifications as suggestions requiring validation |
| Point-in-time assessment | LLM capabilities change over time | Periodic re-testing; framework supports re-assessment |
| MCQ format constraints | Some learning objectives cannot be assessed via MCQ regardless of AI-resistance | Tool can identify such cases |

QVAF cannot:

- Guarantee AI-proof questions
- Replace educator judgment
- Detect AI use after the fact
- Address all cheating behaviours

---

## 6. Conclusion

This literature review supports several design principles for QVAF:

Cognitive complexity serves as a predictor of vulnerability. The relationship between Bloom's Taxonomy levels and LLM performance is supported by empirical evidence (*p* < .05 across multiple models), providing justification for using cognitive classification as a vulnerability indicator.

Regulatory guidance supports redesign over detection. TEQSA, QAA, and academic literature converge on the position that structural assessment change represents a more sustainable approach than AI detection.

LLM unreliability creates both risks and design opportunities. Hallucination rates in educational contexts, overconfidence patterns, and response inconsistency indicate that students using LLMs face risks of receiving incorrect answers, while well-designed questions may exploit these limitations.

Adoption depends on framing and perceived usefulness. Asset-based framing, risk-mitigation messaging, and actionable recommendations are associated with user engagement. TAM research indicates perceived usefulness outweighs ease of use for adoption.

The evidence base continues to evolve as LLM capabilities change. However, the foundations for using cognitive complexity, question format, and context-dependence as vulnerability indicators are established in current literature.

---

## References

Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001). *A taxonomy for learning, teaching, and assessing: A revision of Bloom's taxonomy of educational objectives* (Complete ed.). Longman.

Biggs, J. B., & Collis, K. F. (1982). *Evaluating the quality of learning: The SOLO taxonomy (Structure of the Observed Learning Outcome)*. Academic Press.

Bloom, B. S. (Ed.). (1956). *Taxonomy of educational objectives: The classification of educational goals. Handbook I: Cognitive domain*. David McKay Company.

Brownell, S. E., & Tanner, K. D. (2012). Barriers to faculty pedagogical change: Lack of training, time, incentives, and... tensions with professional identity? *CBE—Life Sciences Education*, *11*(4), 339–346. https://doi.org/10.1187/cbe.12-09-0163

Chhikara, P., Sharma, A., Singla, P., & Krishnamurthy, B. (2025). Mind the confidence gap: Overconfidence, calibration, and distractor effects in large language models. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2502.11028

Corbin, T., Dawson, P., & Liu, D. (2025). Talk is cheap: Why structural assessment changes are needed for a time of GenAI. *Assessment & Evaluation in Higher Education*, *50*(7). https://doi.org/10.1080/02602938.2025.2503964

Council of Regional Accrediting Commissions. (2025, October 6). *Joint statement on artificial intelligence in credit transfer and learning evaluation*. C-RAC.

Davis, F. D. (1989). Perceived usefulness, perceived ease of use, and user acceptance of information technology. *MIS Quarterly*, *13*(3), 319–340. https://doi.org/10.2307/249008

Fong, C. J., Schallert, D. L., Williams, K. M., Williamson, Z. H., Warner, J. R., Lin, S., & Kim, Y. W. (2018). When feedback signals failure but offers hope for improvement: A process model of constructive criticism. *Thinking Skills and Creativity*, *30*, 42–53. https://doi.org/10.1016/j.tsc.2018.02.014

Henderson, C., Beach, A., & Finkelstein, N. (2011). Facilitating change in undergraduate STEM instructional practices: An analytic review of the literature. *Journal of Research in Science Teaching*, *48*(8), 952–984. https://doi.org/10.1002/tea.20439

Herrmann-Werner, A., Festl-Wietek, T., Grunwald, T., Johansson, L., & Zipfel, S. (2024). Assessing ChatGPT's mastery of Bloom's taxonomy using psychosomatic medicine exam questions: Mixed-methods study. *Journal of Medical Internet Research*, *26*, e52113. https://doi.org/10.2196/52113

Huber, M., & Niklaus, J. (2025). LLMs meet Bloom's taxonomy: A cognitive view on large language model evaluations. In *Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025)* (pp. 5234–5251). Association for Computational Linguistics. https://aclanthology.org/2025.coling-main.350/

Khatun, A., & Brown, D. G. (2024). TruthEval: A dataset to evaluate LLM truthfulness and reliability. *arXiv preprint arXiv:2406.01855*. https://arxiv.org/abs/2406.01855

Kizilcec, R. F., Raduescu, C., Kovanovic, V., Joksimovic, S., Cram, A., Smolansky, A., & Zeide, E. (2024). Perceived impact of generative AI on assessments: Comparing educator and student perspectives in Australia, Cyprus, and the United States. *Computers and Education: Artificial Intelligence*, *6*, 100198. https://doi.org/10.1016/j.caeai.2024.100198

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)* (pp. 9459–9474). https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

Li, J., Cheng, X., Zhao, W. X., Nie, J. Y., & Wen, J. R. (2024). The dawn after the dark: An empirical study on factuality hallucination in large language models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)*. Association for Computational Linguistics. https://aclanthology.org/2024.acl-long.586/

Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., & Zou, J. (2023). GPT detectors are biased against non-native English writers. *Patterns*, *4*(7), 100779. https://doi.org/10.1016/j.patter.2023.100779

Liu, S., Okuhara, T., Chang, X., Shirabe, R., Nishiie, Y., Okada, H., & Kiuchi, T. (2024). Performance of ChatGPT across different versions in medical licensing examinations worldwide: Systematic review and meta-analysis. *Journal of Medical Internet Research*, *26*, e60807. https://doi.org/10.2196/60807

Lodge, J. M., Bearman, M., Dawson, P., Gniel, H., Harper, R., Liu, D., McLean, J., Ucnik, L., & Associates. (2025). *Enacting assessment reform in a time of artificial intelligence*. Tertiary Education Quality and Standards Agency. https://www.teqsa.gov.au/

Lodge, J. M., Howard, S., Bearman, M., Dawson, P., & Associates. (2023). *Assessment reform for the age of artificial intelligence*. Tertiary Education Quality and Standards Agency. https://www.teqsa.gov.au/

Ma, W., Liu, S., Wang, Y., & Hu, X. (2025). BloomAPR: A Bloom's taxonomy-based framework for assessing the capabilities of LLM-powered APR solutions. *arXiv preprint arXiv:2509.25465*. https://arxiv.org/abs/2509.25465

Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., & Ho, D. E. (2024). Hallucination-free? Assessing the reliability of leading AI legal research tools. *Stanford Law School Working Paper*. https://dho.stanford.edu/wp-content/uploads/Legal_RAG_Hallucinations.pdf

McDonald, N., Johri, A., Ali, A., & Hingle, A. (2024). *Apostles, agnostics and atheists: Engagement with generative AI by Australian university staff*. QUT Centre for Decent Work and Industry. https://eprints.qut.edu.au/252079/

Myrzakhan, A., Bsharat, S. M., & Shen, Z. (2024). Open-LLM-Leaderboard: From multi-choice to open-style questions for LLMs evaluation, benchmark, and arena. *arXiv preprint arXiv:2406.07545*. https://arxiv.org/abs/2406.07545

Newton, P., & Xiromeriti, M. (2024). ChatGPT performance on multiple choice question examinations in higher education: A pragmatic scoping review. *Assessment & Evaluation in Higher Education*, *49*(6), 781–798. https://doi.org/10.1080/02602938.2023.2299059

Nguyen, T., Tran, H., Le, M., & Pham, K. (2025). Accuracy of latest large language models in answering multiple choice questions in dentistry: A comparative study. *PLOS ONE*, *20*(1), e0317423. https://doi.org/10.1371/journal.pone.0317423

OpenAI. (2024). *Introducing SimpleQA*. https://openai.com/index/introducing-simpleqa/

Ouyang, S., Zhang, J., Wen, M., Zhang, L., Wang, Y., & Guo, Y. (2025). Non-determinism of "deterministic" LLM settings: An empirical study. *ACM Transactions on Software Engineering and Methodology*. https://doi.org/10.1145/3702987

Perkins, M., Furze, L., Roe, J., & MacVaugh, J. (2024a). The Artificial Intelligence Assessment Scale (AIAS): A framework for ethical integration of generative AI in educational assessment. *Journal of University Teaching and Learning Practice*, *21*(6). https://doi.org/10.53761/1.21.6.02

Perkins, M., Roe, J., & Furze, L. (2024b). Revised AI Assessment Scale: A framework for AI in education. *Journal of University Teaching and Learning Practice*, *21*(8). https://doi.org/10.53761/1.21.8.14

Quality Assurance Agency for Higher Education. (2023). *Reconsidering assessment for the ChatGPT era*. QAA. https://www.qaa.ac.uk/docs/qaa/members/reconsidering-assessment-for-the-chat-gpt-era.pdf

Sallam, M., Al-Salahat, K., Almhdawi, K., Eid, H., Ismail, I. I., & Al-Salahat, K. (2024). Exploring the performance of ChatGPT versions 3.5, 4, and 4 with vision in the radiology board-style examination using text- and image-based questions. *Advances in Medical Education and Practice*, *15*, 433–443. https://doi.org/10.2147/AMEP.S463820

Scherer, R., Siddiq, F., & Tondeur, J. (2019). The technology acceptance model (TAM): A meta-analytic structural equation modeling approach to explaining teachers' adoption of digital technology in education. *Computers & Education*, *128*, 13–35. https://doi.org/10.1016/j.compedu.2018.09.009

Schneier, B. (2003). *Beyond fear: Thinking sensibly about security in an uncertain world*. Copernicus Books.

Southern Association of Colleges and Schools Commission on Colleges. (2024, December). *Artificial intelligence in accreditation*. SACSCOC.

Tertiary Education Quality and Standards Agency. (2024). *Risk assessment framework*. TEQSA. https://www.teqsa.gov.au/guides-resources/resources/corporate-publications/risk-assessment-framework

Thinking Machines Lab. (2025, September). When LLMs surprise you: 1,000 identical prompts, 80 different answers. *Thinking Machines Data Science Blog*.

Venkatesh, V., Morris, M. G., Davis, G. B., & Davis, F. D. (2003). User acceptance of information technology: Toward a unified view. *MIS Quarterly*, *27*(3), 425–478. https://doi.org/10.2307/30036540

Walters, W. H., & Wilanda, E. I. (2023). Fabrication and errors in the bibliographic citations generated by ChatGPT. *Scientific Reports*, *13*, 14045. https://doi.org/10.1038/s41598-023-41032-5

Webb, N. L. (1997). *Criteria for alignment of expectations and assessments in mathematics and science education* (Research Monograph No. 6). National Institute for Science Education, University of Wisconsin-Madison.

Weber-Wulff, D., Anohina-Naumeca, A., Bjelobaba, S., Foltýnek, T., Guerrero-Dib, J., Popoola, O., Šigut, P., & Waddington, L. (2023). Testing of detection tools for AI-generated text. *International Journal for Educational Integrity*, *19*(1), 26. https://doi.org/10.1007/s40979-023-00146-z

Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Hooi, B. (2024). Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. In *Proceedings of the Twelfth International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=gjeQKFxFpZ

Xue, L., Rashid, A. M., & Ouyang, S. (2024). The Unified Theory of Acceptance and Use of Technology (UTAUT) in higher education: A systematic review. *SAGE Open*, *14*(1). https://doi.org/10.1177/21582440241229570

Yaacoub, C., Assaghir, Z., & Da-Rugna, J. (2025). Cognitive depth enhancement in AI-driven educational tools via SOLO taxonomy. In *Proceedings of the International Conference on Advanced Computing Research (ACR 2025)* (pp. 15–28). Springer. https://doi.org/10.1007/978-3-031-87647-9_2
