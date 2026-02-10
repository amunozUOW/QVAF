# Literature Review: Theoretical Foundations for a Quiz Vulnerability Assessment Framework

## Introduction
---
Unproctored online quizzes and tests face validity threats from student cheating behaviour and integrity concerns have been compounded by the availability of generative AI tools. Regulatory reports and scholarly research relevant to the design of assessment tools provide guidance towards evaluating vulnerability of online quizzes to misuse of AI in educational settings. The review addresses five domains: regulatory frameworks for assessment design in AI contexts, empirical patterns in large language model (LLM) performance by question type and cognitive level, research on LLM reliability and hallucination, and design rationale for the Quiz Vulnerability Assessment Framework (QVAF). The synthesis lends support to the utility of assessing and addressing online quiz vulnerability as a complementary activity to structural assessment redesign. Collectively these actions are a more sustainable approach than AI detection or prohibition. In essence, AI-powered tools—including LLM-based item quality evaluation can prove valuable to educators seeking to maintain or restore assessment validity. This document is a work in progress and should not be cited.

---
## 1. Regulatory Frameworks for Assessment Redesign
---
In Australia the Tertiary Education Quality and Standards Agency (TEQSA) issued two guidance documents on assessment practices in the context of generative AI. The first document, titled *Assessment Reform for the Age of Artificial Intelligence* articulated two principles: that assessment should prepare students for participation in contexts where AI is prevalent, and that trustworthy judgments about learning require multiple approaches to assessment (Lodge et al., 2023). The document positioned detection as insufficient as a standalone strategy, advocating instead for assessment redesign. These notions are echoed in the scholarly literature. For instance, Corbin et al. (2025) argued that frameworks relying on student compliance remain limited in their capacity to prevent AI use. Similarly, Perkins et al., (2024) acknowledged that permitting AI use in assessments creates challenges for enforcement due to detection limitations. Such a notion resonates with the work of Kizilcec et al. (2024), who reported educators stated a preference for assessments adapted to assume AI use. Subsequently to the TEQSA 2023 guidance, Lodge et al. (2025) published *Enacting Assessment Reform in a Time of Artificial Intelligence*, which outlined three implementation pathways: program-wide reform treating assessment as a connected system across degree programs; unit-level assurance embedding at least one secure assessment within each unit; and hybrid approaches combining both strategies. Importantly, this later guidance points to approaches that "design assessments where gen AI use is irrelevant to the demonstration of learning outcomes." This notion provides an important underpinning to this research. 

---
## 2. Cognitive Complexity and LLM Performance
---
Empirical research indicates that LLMs perform differently across cognitive levels, with performance patterns varying by question type and format.  Newton and Xiromeriti (2024) conducted a scoping review of ChatGPT performance on multiple-choice question (MCQ) examinations and found that GPT-3 and GPT-3.5 performed above chance but failed most examinations, while GPT-4 passed most examinations with performance comparable to human subjects. Sallam et al. (2024) examined ChatGPT performance on radiology board-style examinations and found statistically significant differences in performance across cognitive domains, with higher performance on Remember and Understand levels compared to Apply and Analyze levels.

Huber and Niklaus (2025) analysed LLM performance in relation to Bloom's Taxonomy and found that LLMs performed better on lower levels of Bloom's Taxonomy and identified gaps in benchmark coverage of higher cognitive skills. Operationalising such guidance in an educational assessment context would tend to suggest that unproctored assessments, such as online quizzes, may be less vulnerable to AI misuse if questions are targeted at evaluating higher order cognitive skills. Unfortunately, research on LLM performance across Webb's Depth of Knowledge levels remains limited. Most existing DOK research has focused on using LLMs to generate DOK-aligned questions rather than testing performance against them.

In terms of question formats and features, Nguyen et al. (2025) found LLM performance decreased when tested against image-based questions. Similarly, Myrzakhan et al. (2024) demonstrated that open-style questions lowered LLM accuracy relative to MCQ accuracy across the models tested, and LLMs exhibited selection bias toward certain option positions.

---
## 3. LLM Reliability and Hallucination
---
Research on LLM reliability identifies patterns relevant to assessment vulnerability, including overconfidence in incorrect responses, hallucination across domains, and response inconsistency. Xiong et al. (2024) examined confidence elicitation in LLMs and found that LLMs expressed overconfidence when verbalising confidence levels, with values predominantly in the 80–100% range in multiples of five. Calibration improved with model capability but remained imperfect. Similarly, Chhikara et al. (2025) found that incorporating distractors achieved relative accuracy improvements and increased miscalibration on easier queries.

Ouyang et al. (2025) examined non-determinism in LLM outputs and found that LLMs produce different responses even at temperature = 0 due to batching, floating-point operation ordering, and GPU parallelism effects. Thinking Machines Lab (2025) demonstrated this pattern by submitting 1,000 identical prompts and receiving 80 different responses, with divergence occurring after approximately 100 tokens. Khatun and Brown (2024) found that models showed vulnerability to prompt variations, with performance gaps across different phrasings of identical queries. Such variability in responses, despite setting temperature to zero will be inherent in the QVAF as well as in settings where students misuse genAI. However, student use of AI may not occur under temperature=0 settings and higher variability may be observed. For the time being, the QVAF temperature is fixed at zero to minimise stochasticity, especially across repeated observations.

---
### 3.1 Retrieval-Augmented Generation
---
Lewis et al. (2020) introduced Retrieval-Augmented Generation (RAG) for knowledge-intensive NLP tasks at NeurIPS 2020, establishing an architecture for grounding LLM responses in retrieved documents. However, the use of RAG systems does not guarantee an elimination of hallucinations. Magesh et al. (2024) evaluated legal RAG systems and found that systems hallucinated 17–33% of responses. The authors concluded that vendor claims regarding hallucination-free systems were not supported by empirical evidence.

Yet, research on domain-specific RAG effects indicates reductions in hallucination rates; in biomedical contexts, hallucination rates decreased from approximately 49% to 24% with RAG implementation. As such, it is recommended that users construct RAGs with course content materials that are verified as relevant to the quizzes being evaluated.

---
## 4. Educator Adoption Considerations for QVAF
---
Research on technology acceptance and educator attitudes provides guidance for designing assessment tools that educators will adopt. In terms of technology adoption, and a possible commentary on the adoption of the QVAF, we rely on existing theories on technology adoption. Primarily, we rely on Davis (1989), who established the Technology Acceptance Model (TAM), finding that Perceived Usefulness had stronger correlation with usage behaviour than Perceived Ease of Use. Scherer et al. (2019) found that the model explained technology acceptance with variation across contexts, and that facilitating conditions (training, technical support, institutional backing) affected adoption.

### 4.1 Risk-Based Approaches and Assessment Validity

TEQSA (2024) published a Risk Assessment Framework stating that "TEQSA recognises that innovation often involves a degree of risk taking and does not consider risk as necessarily negative or that all risk must be controlled or eliminated." The academic integrity threat can also be framed in terms of assessment validity. Messick (1989) defined assessment validity as an integrated evaluative judgment of the degree to which empirical evidence and theoretical rationales support the adequacy and appropriateness of inferences and actions based on test scores. Cheating presents barriers to the certification of student learning objective attainment, and a threat to the validity of the assessment (Dawson et al., 2024). Assessment validity threats are not new, research conducted prior to 2023 documented validity threats in unproctored online assessments arising from unauthorised resource use, collaboration, and identity fraud. However, such threats have increased in scale in the age of genAI (Newton, 2025), 

Schneier (2003) introduced the concept of security theatre, referring to measures designed to create impressions of safety rather than actual security. To this end, the QVAF has been designed to provide constructive feedback, framed from the premise that objective measures of vulnerability provide an assessment of risk, but do not prescribe redesigns of individual questions. This notion echoes Fong et al. (2018) model of constructive criticism, where criticism is perceived as constructive when it identifies gaps and provides directions for improvement. Further support for this approach exists in research on asset-based versus deficit-based framing indicates that deficit framing leads to disengagement, while asset-based framing supports motivation.

### 4.2 Barriers to Pedagogical Change

Henderson et al. (2011) reviewed literature on change in undergraduate STEM instructional practices and found that faculty development efforts that treated instructional change as a knowledge problem rather than a cultural and identity problem were less effective. Further barriers to adoption were found by Brownell and Tanner (2012), who identified barriers to faculty pedagogical change including lack of training, time, and incentives. Instructional change can lead to lower teaching evaluations when students resist change.

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

QVAF employs a five-level cognitive demand taxonomy synthesising Bloom's Revised Taxonomy (Bloom, 1956, Anderson & Krathwohl, 2001), Webb's Depth of Knowledge (Webb, 1997), and SOLO Taxonomy (Biggs & Collis, 1982):

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

Cognitive complexity serves as a predictor of vulnerability. The relationship between Bloom's Taxonomy levels and LLM performance is supported by empirical evidence, providing justification for using cognitive classification as a vulnerability indicator.

Regulatory guidance supports redesign over detection. TEQSA and academic literature converge on the position that structural assessment change represents a more sustainable approach than AI detection.

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

Davis, F. D. (1989). Perceived usefulness, perceived ease of use, and user acceptance of information technology. *MIS Quarterly*, *13*(3), 319–340. https://doi.org/10.2307/249008

Dawson, P., Bearman, M., Dollinger, M., & Boud, D. (2024). Validity matters more than cheating. Assessment; Evaluation in Higher Education, 49(7), 1005–1016. https://doi.org/10.1080/02602938.2024.2386662

Fong, C. J., Schallert, D. L., Williams, K. M., Williamson, Z. H., Warner, J. R., Lin, S., & Kim, Y. W. (2018). When feedback signals failure but offers hope for improvement: A process model of constructive criticism. *Thinking Skills and Creativity*, *30*, 42–53. https://doi.org/10.1016/j.tsc.2018.02.014

Henderson, C., Beach, A., & Finkelstein, N. (2011). Facilitating change in undergraduate STEM instructional practices: An analytic review of the literature. *Journal of Research in Science Teaching*, *48*(8), 952–984. https://doi.org/10.1002/tea.20439

Huber, M., & Niklaus, J. (2025). LLMs meet Bloom's taxonomy: A cognitive view on large language model evaluations. In *Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025)* (pp. 5234–5251). Association for Computational Linguistics. https://aclanthology.org/2025.coling-main.350/

Khatun, A., & Brown, D. G. (2024). TruthEval: A dataset to evaluate LLM truthfulness and reliability. *arXiv preprint arXiv:2406.01855*. https://arxiv.org/abs/2406.01855

Kizilcec, R. F., Raduescu, C., Kovanovic, V., Joksimovic, S., Cram, A., Smolansky, A., & Zeide, E. (2024). Perceived impact of generative AI on assessments: Comparing educator and student perspectives in Australia, Cyprus, and the United States. *Computers and Education: Artificial Intelligence*, *6*, 100198. https://doi.org/10.1016/j.caeai.2024.100198

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)* (pp. 9459–9474). https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

Lodge, J. M., Bearman, M., Dawson, P., Gniel, H., Harper, R., Liu, D., McLean, J., Ucnik, L., & Associates. (2025). *Enacting assessment reform in a time of artificial intelligence*. Tertiary Education Quality and Standards Agency. https://www.teqsa.gov.au/

Lodge, J. M., Howard, S., Bearman, M., Dawson, P., & Associates. (2023). *Assessment reform for the age of artificial intelligence*. Tertiary Education Quality and Standards Agency. https://www.teqsa.gov.au/

Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., & Ho, D. E. (2024). Hallucination-free? Assessing the reliability of leading AI legal research tools. *Stanford Law School Working Paper*. https://dho.stanford.edu/wp-content/uploads/Legal_RAG_Hallucinations.pdf

Messick, S. (1989). Validity. In R. L. Linn (Ed.), Educational measurement (3rd ed., pp. 13–103). American Council on Education and Macmillan.

Myrzakhan, A., Bsharat, S. M., & Shen, Z. (2024). Open-LLM-Leaderboard: From multi-choice to open-style questions for LLMs evaluation, benchmark, and arena. *arXiv preprint arXiv:2406.07545*. https://arxiv.org/abs/2406.07545

Newton, P., & Xiromeriti, M. (2024). ChatGPT performance on multiple choice question examinations in higher education: A pragmatic scoping review. *Assessment & Evaluation in Higher Education*, *49*(6), 781–798. https://doi.org/10.1080/02602938.2023.2299059

Newton, P. M. (2025). How vulnerable are UK universities to cheating with new GenAI tools? A pragmatic risk assessment. Assessment; Evaluation in Higher Education, 50(8), 1332–1343. https://doi.org/10.1080/02602938.2025.2511794

Nguyen, T., Tran, H., Le, M., & Pham, K. (2025). Accuracy of latest large language models in answering multiple choice questions in dentistry: A comparative study. *PLOS ONE*, *20*(1), e0317423. https://doi.org/10.1371/journal.pone.0317423

Ouyang, S., Zhang, J., Wen, M., Zhang, L., Wang, Y., & Guo, Y. (2025). Non-determinism of "deterministic" LLM settings: An empirical study. *ACM Transactions on Software Engineering and Methodology*. https://doi.org/10.1145/3702987

Perkins, M., Roe, J., & Furze, L. (2024). Revised AI Assessment Scale: A framework for AI in education. *Journal of University Teaching and Learning Practice*, *21*(8). https://doi.org/10.53761/1.21.8.14

Sallam, M., Al-Salahat, K., Almhdawi, K., Eid, H., Ismail, I. I., & Al-Salahat, K. (2024). Exploring the performance of ChatGPT versions 3.5, 4, and 4 with vision in the radiology board-style examination using text- and image-based questions. *Advances in Medical Education and Practice*, *15*, 433–443. https://doi.org/10.2147/AMEP.S463820

Scherer, R., Siddiq, F., & Tondeur, J. (2019). The technology acceptance model (TAM): A meta-analytic structural equation modeling approach to explaining teachers' adoption of digital technology in education. *Computers & Education*, *128*, 13–35. https://doi.org/10.1016/j.compedu.2018.09.009

Schneier, B. (2003). *Beyond fear: Thinking sensibly about security in an uncertain world*. Copernicus Books.

Tertiary Education Quality and Standards Agency. (2024). *Risk assessment framework*. TEQSA. https://www.teqsa.gov.au/guides-resources/resources/corporate-publications/risk-assessment-framework

Thinking Machines Lab. (2025, September). When LLMs surprise you: 1,000 identical prompts, 80 different answers. *Thinking Machines Data Science Blog*.

Webb, N. L. (1997). *Criteria for alignment of expectations and assessments in mathematics and science education* (Research Monograph No. 6). National Institute for Science Education, University of Wisconsin-Madison.

Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Hooi, B. (2024). Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. In *Proceedings of the Twelfth International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=gjeQKFxFpZ

