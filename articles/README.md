# POI Itinerary / Tour Recommendation — Annotated Bibliography

Curated for the **itinerary** phase of the thesis (planning a *sequence/route of multiple POIs* under constraints — distinct from the next-POI prediction baseline in the rest of this repo).

- **81 papers** identified via a verified multi-source literature sweep.
- **46 open-access PDFs downloaded** into this folder (✅), validated as real PDFs.
- Remaining entries are paywalled (🔒) — DOIs provided; you have journal access.
- Two key **dataset/code resources** listed at the end.

Legend: ✅ = PDF in this folder · 🔒 = paywalled (link only) · 🔗 = OA but not auto-downloadable (link only).

---

## How to read this list (suggested path)

If you are coming from next-POI prediction, read in this order:
1. **Surveys** → Lim 2019 (KAIS) and Halder 2024 (ASOC) frame the whole field.
2. **OR foundations** → Golden 1987, Vansteenwegen 2011 survey, PersTour 2015/2018 (the orienteering-problem formulation).
3. **Learning-based bridge** → Chen 2016 "Learning Points and Routes" + 2017 "Structured Recommendation" (introduces the order-aware *pairs-F1* metric reused everywhere).
4. **Deep learning** → DeepTrip 2019, SelfTrip 2022, GETNext 2022.
5. **LLM era** → TravelPlanner 2024, ItiNera 2024, then the 2025 benchmarks.

---

## 1. Orienteering Problem & TTDP foundations (operations-research era)

The Orienteering Problem (OP) and its variants (Team OP, OP with Time Windows) are the optimization backbone of classic itinerary recommendation.

- 🔒 **The orienteering problem** — Golden, Levy, Vohra. *Naval Research Logistics 1987*. [DOI](https://doi.org/10.1002/1520-6750(198706)34:3%3C307::AID-NAV3220340302%3E3.0.CO;2-D). The seminal OP formalization (max-score node subset within a budget).
- 🔒 **Heuristic methods applied to orienteering** — Tsiligirides. *JORS 1984*. [DOI](https://doi.org/10.1057/jors.1984.162). Original S-/D-algorithm heuristics; historical root of OP heuristics.
- 🔒 **The team orienteering problem** — Chao, Golden, Wasil. *EJOR 1996*. [DOI](https://doi.org/10.1016/0377-2217(94)00289-4). TOP = multiple routes/days; basis for multi-day itineraries.
- ✅ `2009_Souffriau_ILS-TOPTW_COR.pdf` — **Iterated local search for the TOP with time windows** — Souffriau et al. *Computers & OR 2009*. [DOI](https://doi.org/10.1016/j.cor.2009.03.008). Fast ILS metaheuristic; core engine of multi-POI tour generation.
- 🔒 **A personalized TTDP algorithm for mobile tourist guides** — Souffriau, Vansteenwegen et al. *Applied Artificial Intelligence 2008*. [DOI](https://doi.org/10.1080/08839510802379626). Early OP-based personalized mobile itineraries.
- 🔒 **The City Trip Planner: an expert system for tourists** — Vansteenwegen et al. *ESWA 2011*. [DOI](https://doi.org/10.1016/j.eswa.2010.11.085). Deployed planner: VSM POI scores + TOPTW solved with GRASP.
- ✅ `2015-2018_Lim_PersTour_KAIS-extended.pdf` — **PersTour: Personalized Tour Recommendation based on User Interests and POI Visit Durations** — Lim, Chan, Leckie, Karunasekera. *IJCAI 2015* ([abs](https://www.ijcai.org/Abstract/15/253)); extended in *KAIS 2018* ([DOI](https://doi.org/10.1007/s10115-017-1056-y)). **The** seminal personalized OP itinerary method; the downloaded PDF is the KAIS journal version (adds visit-recency).
- ✅ `2016_Lim_PersonalizedItineraries_UMAP.pdf` — **Personalized Recommendation of Travel Itineraries...** — Lim. *UMAP 2016 (DC)*. Start/end-POI + time-budget constrained individual & group itineraries.
- ✅ `2018_Taylor_TourMustSee_LocWeb.pdf` — **Travel Itinerary Recommendations with Must-see POIs** — Taylor, Lim, Chan. *WWW Companion / LocWeb 2018*. ILP variant of OP enforcing must-visit POIs.
- 🔒 **Personalized Itinerary Recommendation with Queuing Time Awareness (PersQ)** — Lim et al. *SIGIR 2017*. [DOI](https://doi.org/10.1145/3077136.3080778). OP + time-varying queue times.
- 🔒 **Planning the trip itinerary for tourist groups** — Sylejmani, Dorn, Musliu. *Inf. Technology & Tourism 2017*. [DOI](https://doi.org/10.1007/s40558-017-0080-9). Multi-constrained team-OP for groups.
- 🔒 **Customized tour recommendations in urban areas** — Gionis, Lappas, Pelechrinis, Terzi. *WSDM 2014*. [DOI](https://doi.org/10.1145/2556195.2559893). Constrained route selection over POI categories.
- ✅ `2025_PlusTour-SmartTourism_ComputerNetworks.pdf` — **+Tour: Recommending Personalized Itineraries for Smart Tourism** — Espera et al. *Computer Networks 2025* / [arXiv:2502.17345](https://arxiv.org/abs/2502.17345). OP-style itinerary + edge-resource allocation, exact algorithm.

---

## 2. Photo / breadcrumb-driven itinerary construction (early data-mining)

- ✅ `2010_DeChoudhury_SocialBreadcrumbs_HT-WWW.pdf` — **Automatic construction of travel itineraries using social breadcrumbs** — De Choudhury et al. *Hypertext 2010* ([DOI](https://doi.org/10.1145/1810617.1810626)). Seminal: mine geo-temporal Flickr breadcrumbs → POI sequences + dwell times.
- ✅ `2010_Lu_Photo2Trip_ACMMM.pdf` — **Photo2Trip: Generating Travel Routes from Geo-Tagged Photos** — Lu et al. *ACM MM 2010*. [DOI](https://doi.org/10.1145/1873951.1873972). Routes from ~20M photos under user constraints.
- 🔒 **Travel route recommendation using geotags in photo sharing sites** — Kurashima, Iwata, Irie, Fujimura. *CIKM 2010*. [DOI](https://doi.org/10.1145/1871437.1871513). Topic model (preference) + Markov model (transitions) → personalized routes.
- 🔒 **Personalized travel recommendation by mining people attributes from community photos** — Cheng et al. *ACM MM 2011*. [DOI](https://doi.org/10.1145/2072298.2072311). Demographic-attribute-aware photo-driven travel rec.
- 🔒 **TripBuilder: A Tool for Recommending Sightseeing Tours** — Brilhante et al. *ECIR 2014*. [DOI](https://doi.org/10.1007/978-3-319-06028-6_93). Time-budgeted tours from Wikipedia POIs + Flickr movement.
- 🔒 **On planning sightseeing tours with TripBuilder** — Brilhante et al. *Information Processing & Management 2015*. [DOI](https://doi.org/10.1016/j.ipm.2014.07.008). Journal version: TripCover + Trajectory Scheduling.

---

## 3. Classical ML / structured prediction / learning-to-rank for trajectory rec

- ✅ `2016_Chen_LearningPointsRoutes_CIKM.pdf` — **Learning Points and Routes to Recommend Trajectories** — Chen, Ong, Xie. *CIKM 2016* / [arXiv:1608.07051](https://arxiv.org/abs/1608.07051). Bridges OR and learning; introduces order-aware **pairs-F1** metric.
- ✅ `2017_Chen_StructuredRecommendation.pdf` — **Structured Recommendation** — Chen, Xie, Menon, Ong. [arXiv:1706.09067](https://arxiv.org/abs/1706.09067). Trajectory rec as structured SVM with multiple ground truths + loop-free inference.
- ✅ `2017_Menon_RevisitingRevisits-TrajRec.pdf` — **Revisiting revisits in trajectory recommendation** — Menon, Chen, Xie, Ong. *CitRec@RecSys 2017* / [arXiv:1708.05165](https://arxiv.org/abs/1708.05165). Loop-free sequence inference (ILP, list-Viterbi).
- ✅ `2017_He_CategoryAware-ListwiseBPR_IJCAI.pdf` — **Category-aware Next-POI Recommendation via Listwise BPR** — He, Li, Liao. *IJCAI 2017*. [PDF](https://www.ijcai.org/proceedings/2017/0255.pdf). Third-order tensor + Plackett-Luce listwise ranking.
- ✅ `2019_Cui_Distance2Pre_PAKDD.pdf` — **Distance2Pre: Personalized Spatial Preference for Next-POI** — Cui et al. *PAKDD 2019*. Sequential + spatial (inter-POI distance) preference fusion.
- ✅ `2018_He_JointContextAwareEmbedding-Trip_ICDE.pdf` — **A Jointly Learned Context-Aware Place Embedding for Trip Recommendations** — He, Qi, Ramamohanarao. *ICDE 2019* / [arXiv:1808.08023](https://arxiv.org/abs/1808.08023). POI embeddings capturing popularity + co-occurrence + preference.

---

## 4. Deep learning (RNN / seq2seq / RL / adversarial) trip recommendation

- ✅ `2019_Gao_DeepTrip_SIGSPATIAL.pdf` — **DeepTrip: Adversarially Understanding Human Mobility for Trip Recommendation** — Gao et al. *SIGSPATIAL 2019*. RNN encoder-decoder + adversarial latent regularization.
- ✅ `2019_Wang_NeuralAstar-RouteRec_KDD.pdf` — **Empowering A\* Search with Neural Networks for Personalized Route Recommendation** — Wang et al. *KDD 2019* / [arXiv:1907.08489](https://arxiv.org/abs/1907.08489). Learned cost function + A* search.
- ✅ `2019_Liao_DRPS-POISequence_IJGI.pdf` — **Dynamic Recommendation of POI Sequence (DRPS)** — Liao et al. *ISPRS IJGI 2019* (MDPI, OA). BiLSTM + attention seq2seq for POI sequences.
- ✅ `2021_Jiang_AdversarialNeuralTrip.pdf` — **Adversarial Neural Trip Recommendation (ANT)** — Jiang et al. [arXiv:2109.11731](https://arxiv.org/abs/2109.11731). Attention encoder-decoder + adversarial + RL.
- ✅ `2021_Rashid_DeepAltTrip_TKDE.pdf` — **DeepAltTrip: Top-k Alternative Itineraries** — Rashid, Ali, Cheema. *IEEE TKDE* / [arXiv:2109.03535](https://arxiv.org/abs/2109.03535). Graph autoencoder + fwd/bwd LSTM (ITRNet) + route sampler.
- 🔒 **Semi-supervised Trajectory Understanding with POI Attention (TRED)** — Zhou et al. *ACM TSAS 2020*. [DOI](https://doi.org/10.1145/3378890). Early end-to-end seq2seq trip rec baseline.
- 🔒 **Itinerary Planning via Deep Reinforcement Learning** — Zhu et al. *ICMR 2020*. [DOI](https://doi.org/10.1145/3372278.3390727). MDP + variational agent + DQN.
- 🔒 **Trip Reinforcement Recommendation with Graph-based Representation Learning** — Wu et al. *ACM TKDD 2022*. [DOI](https://doi.org/10.1145/3564609). GNN + RL trip generation.
- 🔒 **Multi-objective reinforcement learning approach for trip recommendation (MORL-Trip)** — Chen et al. *ESWA 2023*. [DOI](https://doi.org/10.1016/j.eswa.2023.120145). Multi-objective actor-critic over POI sequences.

---

## 5. GNN / transformer / self-supervised itinerary recommendation

- ✅ `2022_Gao_SelfTrip_KBS.pdf` — **Self-supervised Representation Learning for Trip Recommendation (SelfTrip)** — Gao et al. *Knowledge-Based Systems 2022* / [arXiv:2109.00968](https://arxiv.org/abs/2109.00968). Two-step contrastive SSL + trip augmentations.
- ✅ `2022_Ho_POIBERT_BigData.pdf` — **POIBERT: A Transformer-based Model for the Tour Recommendation Problem** — Ho, Lim. *IEEE BigData 2022* / [arXiv:2212.13900](https://arxiv.org/abs/2212.13900). Masked-LM sequence learning for tours.
- ✅ `2021_Ho_TourRec-POIEmbedding_IUI.pdf` — **User Preferential Tour Recommendation Based on POI-Embedding Methods** — Ho, Lim. *IUI 2021 Companion* / [arXiv:2103.02464](https://arxiv.org/abs/2103.02464). Skip-Gram/CBOW/FastText POI embeddings + iterative itinerary gen.
- ✅ `2023_Ho_BTRec_RecTour.pdf` — **BTRec: BERT-Based Trajectory Recommendation for Personalized Tours** — Ho, Lee, Lim. *RecTour@RecSys 2023* / [arXiv:2310.19886](https://arxiv.org/abs/2310.19886). POIBERT + user demographics.
- ✅ `2023_Ho_SBTRec_BigData.pdf` — **SBTRec: Transformer + Sentiment Analysis for Tour Recommendation** — Ho, Lee, Lim. *IEEE BigData 2023* / [arXiv:2311.11071](https://arxiv.org/abs/2311.11071).
- ✅ `2023_Ho_LanguageModels-TourItinerary_IJCAI-PMAI.pdf` — **Utilizing Language Models for Tour Itinerary Recommendation** — Ho, Lim. *PMAI@IJCAI 2023* / [arXiv:2311.12355](https://arxiv.org/abs/2311.12355).
- ✅ `2022_Yang_GETNext_SIGIR.pdf` — **GETNext: Trajectory Flow Map Enhanced Transformer for Next POI** — Yang, Liu, Zhao. *SIGIR 2022* / [arXiv:2303.04741](https://arxiv.org/abs/2303.04741). Global trajectory-flow graph fused into a transformer (also a key next-POI SOTA baseline).
- ✅ `2024_Shu_ARTrip-Repetitions_SIGIR.pdf` — **Analyzing and Mitigating Repetitions in Trip Recommendation (AR-Trip)** — Shu et al. *SIGIR 2024* / [arXiv:2507.19798](https://arxiv.org/abs/2507.19798). Cycle-aware predictor to avoid duplicate POIs.
- ✅ `2025_Halder_DLIR-DynamicPOI_TORS.pdf` — **Deep Learning of Dynamic POI Generation and Optimisation for Itinerary Recommendation (DLIR)** — Halder, Lim, Chan, Zhang. *ACM TORS 2025*. Transformer (temporal interest) + GCN (spatial co-visit); recent SOTA.
- 🔒 **Contrastive Trajectory Learning for Tour Recommendation (CTLTR)** — Zhou et al. *ACM TIST 2021*. [DOI](https://doi.org/10.1145/3461617). Self-supervised contrastive tour rec; common baseline.
- 🔒 **GC-TripRec: Graph contextualized generative network with adversarial learning** — Chen et al. *WWW Journal 2023*. [DOI](https://doi.org/10.1007/s11280-022-01127-x). Transition graph + GCN + LSTM, query-aligned.
- 🔒 **Query2Trip: Dual-Debiased Learning for Neural Trip Recommendation** — Wang et al. *DASFAA 2023*. [DOI](https://doi.org/10.1007/978-3-031-30672-3_6). Adversarial + contrastive debiasing.
- 🔒 **Dual-grained human mobility learning ... spatial-temporal graph knowledge fusion** — Gao et al. *Information Fusion 2023*. [DOI](https://doi.org/10.1016/j.inffus.2022.08.022). ST POI graph + coarse/fine mobility.
- 🔒 **Self-supervised contrastive learning for itinerary recommendation** — *ESWA 2024*. [DOI](https://doi.org/10.1016/j.eswa.2024.125683). SCL for sparsity + noisy implicit feedback.

---

## 6. LLM-based travel / itinerary planning & benchmarks (2024–2026)

- ✅ `2024_Xie_TravelPlanner_ICML.pdf` — **TravelPlanner: A Benchmark for Real-World Planning with Language Agents** — Xie et al. *ICML 2024 (Spotlight)* / [arXiv:2402.01622](https://arxiv.org/abs/2402.01622). The reference benchmark (1,225 intents, ~4M records).
- ✅ `2024_Tang_ItiNera_EMNLP.pdf` — **ItiNera: Integrating Spatial Optimization with LLMs for Open-domain Urban Itinerary Planning** — Tang et al. *EMNLP 2024 Industry* / [arXiv:2402.07204](https://arxiv.org/abs/2402.07204). LLM POI selection + cluster-aware spatial optimization. **Most directly on-topic.**
- ✅ `2024_Ju_ToTheGlobe-TTG_EMNLP-Demo.pdf` — **To the Globe (TTG): Language-Driven Guaranteed Travel Planning** — Ju et al. *EMNLP 2024 Demo* / [arXiv:2410.16456](https://arxiv.org/abs/2410.16456). LLM → MILP solver → guaranteed-optimal itineraries.
- ✅ `2025_Hao_LLM-FormalVerification_NAACL.pdf` — **LLMs Can Solve Real-World Planning Rigorously with Formal Verification Tools** — Hao et al. *NAACL 2025* / [arXiv:2404.11891](https://arxiv.org/abs/2404.11891). SMT-based; raises TravelPlanner success ~10%→93.9%.
- ✅ `2024_Chen_TravelAgent.pdf` — **TravelAgent: An AI Assistant for Personalized Travel Planning** — Chen et al. [arXiv:2409.08069](https://arxiv.org/abs/2409.08069). Tool-use + planning + memory modules.
- ✅ `2025_Wang_Vaiage-MultiAgent.pdf` — **Vaiage: A Multi-Agent Solution to Personalized Travel Planning** — Wang et al. [arXiv:2505.10922](https://arxiv.org/abs/2505.10922). Route/strategy agents over attraction sequences.
- ✅ `2025_Chaudhuri_TripCraft_ACL.pdf` — **TripCraft: A Benchmark for Spatio-Temporally Fine-Grained Travel Planning** — Chaudhuri et al. *ACL 2025* / [arXiv:2502.20508](https://arxiv.org/abs/2502.20508). Transit/events/personas + continuous metrics.
- ✅ `2025_Shen_TripTailor_ACL-Findings.pdf` — **TripTailor: A Real-World Benchmark for Personalized Travel Planning** — Shen et al. *Findings of ACL 2025* / [arXiv:2508.01432](https://arxiv.org/abs/2508.01432). 500k+ POIs, ~4,000 itineraries.
- ✅ `2024_Shao_ChinaTravel.pdf` — **ChinaTravel: Open-Ended Travel Planning Benchmark w/ Compositional Constraint Validation** — Shao et al. *ICLR 2026* / [arXiv:2412.13682](https://arxiv.org/abs/2412.13682). 372 cities, 5,670 attractions; neuro-symbolic agents + DSL.
- ✅ `2025_Ni_TP-RAG_EMNLP.pdf` — **TP-RAG: Benchmarking Retrieval-Augmented LLM Agents for Spatiotemporal-Aware Travel Planning** — Ni et al. *EMNLP 2025* / [arXiv:2504.08694](https://arxiv.org/abs/2504.08694). First RAG benchmark (2,348 queries, 85k POIs) + EvoRAG.
- ✅ `2025_TripTide_AdaptivePlanning.pdf` — **TripTide: A Benchmark for Adaptive Travel Planning under Disruptions** — Karmakar et al. [arXiv:2510.21329](https://arxiv.org/abs/2510.21329). Dynamic itinerary repair / replanning.
- ✅ `2024_Ren_LLMsReady-TravelPlanning.pdf` — **Are Large Language Models Ready for Travel Planning?** — Ren et al. [arXiv:2410.17333](https://arxiv.org/abs/2410.17333). Bias & hallucination audit.
- ✅ `2026_Revisiting-LLM-TravelPlanning.pdf` — **Revisiting the Travel Planning Capabilities of LLMs** — Zhang et al. 2026 / [arXiv:2605.03308](https://arxiv.org/abs/2605.03308). Decomposes planning into 5 atomic sub-capabilities.

---

## 7. Surveys & taxonomies

- ✅ `2019_Lim_TourRecTripPlanning-Survey_KAIS.pdf` — **Tour recommendation and trip planning using LBSNs: a survey** — Lim et al. *KAIS 2019*. **Best entry point** (data, formulations, algorithms, evaluation, open problems).
- ✅ `2024_Halder_ItineraryRec-Survey_ASOC.pdf` — **A survey on personalized itinerary recommendation: from optimisation to deep learning** — Halder et al. *Applied Soft Computing 2024*. Most current itinerary-specific taxonomy.
- ✅ `2024_Zhang_POIRec-Survey.pdf` — **A Survey on POI Recommendation: Models, Architectures, and Security** — Zhang et al. *2024* / [arXiv:2410.02191](https://arxiv.org/abs/2410.02191). POI rec (the building block), traditional → LLMs.
- 🔗 **A systematic literature review for the tourist trip design problem** — Ruiz-Meza, Montoya-Torres. *Operations Research Perspectives 2022* (OA, CC-BY). [DOI](https://doi.org/10.1016/j.orp.2022.100228) — *open access but scrape-blocked; download via your ScienceDirect access.*
- 🔒 **A survey on algorithmic approaches for solving tourist trip design problems** — Gavalas et al. *Journal of Heuristics 2014*. [DOI](https://doi.org/10.1007/s10732-014-9242-5). Canonical TTDP↔OP mapping.
- 🔒 **The orienteering problem: A survey** — Vansteenwegen, Souffriau, Van Oudheusden. *EJOR 2011*. [DOI](https://doi.org/10.1016/j.ejor.2010.03.045).
- 🔒 **Orienteering Problem: recent variants, solution approaches and applications** — Gunawan, Lau, Vansteenwegen. *EJOR 2016*. [DOI](https://doi.org/10.1016/j.ejor.2016.04.059).
- 🔒 **Mobile recommender systems in tourism** — Gavalas et al. *J. Network & Computer Applications 2014*. [DOI](https://doi.org/10.1016/j.jnca.2013.04.006).
- 🔒 **Intelligent tourism recommender systems: A survey** — Borràs, Moreno, Valls. *ESWA 2014*. [DOI](https://doi.org/10.1016/j.eswa.2014.06.007).
- 🔒 **A Comprehensive Survey on Travel Recommender Systems** — Chaudhari, Thakkar. *Arch. Comput. Methods Eng. 2020*. [DOI](https://doi.org/10.1007/s11831-019-09363-7).
- 🔒 **Tourism recommendation system: a survey and future research directions** — Sarkar et al. *Multimedia Tools & Applications 2022*. [DOI](https://doi.org/10.1007/s11042-022-12167-w).
- 🔒 **Points of Interest recommendations: Methods, evaluation, and future directions** — Werneck et al. *Information Systems 2021*. [DOI](https://doi.org/10.1016/j.is.2021.101789).
- 🔒 **A Survey on POI Recommendation in LBSNs** — Werneck et al. *WebMedia 2020*. [DOI](https://doi.org/10.1145/3428658.3430970).

---

## 8. Methodological backbones (neural combinatorial optimization)

- ✅ `2015_Vinyals_PointerNetworks_NeurIPS.pdf` — **Pointer Networks** — Vinyals, Fortunato, Jaitly. *NeurIPS 2015* / [arXiv:1506.03134](https://arxiv.org/abs/1506.03134). Seq2seq for combinatorial ordering; basis for many route/itinerary models.
- ✅ `2019_Kool_AttentionRoutingProblems_ICLR.pdf` — **Attention, Learn to Solve Routing Problems!** — Kool, van Hoof, Welling. *ICLR 2019* / [arXiv:1803.08475](https://arxiv.org/abs/1803.08475). Attention + REINFORCE for OP/TSP/PCTSP; neural-CO backbone for budget-constrained routing.

---

## 9. Datasets & code resources

- ✅ `2025_Wongso_Massive-STEPS_Dataset.pdf` — **Massive-STEPS: Massive Semantic Trajectories for Understanding POI Check-ins** — Wongso et al. *2025* / [arXiv:2505.11239](https://arxiv.org/abs/2505.11239). 12–15 globally diverse cities; supervised + zero-shot benchmarks. **Code/data:** https://github.com/cruiseresearchgroup/Massive-STEPS
- 🔗 **Lim et al. Flickr-based itinerary datasets** (Toronto, Osaka, Glasgow, Edinburgh, Melbourne, Vienna, Perth, Budapest) — the standard benchmark for classical & deep tour recommendation. **Data/code:** https://sites.google.com/site/limkwanhui/datacode
- 🔒 **Improving Personalized Trip Recommendation to Avoid Crowds Using Pedestrian Sensor Data** — Wang et al. *CIKM 2016*. [DOI](https://doi.org/10.1145/2983323.2983749). Source of the public Melbourne User-POI dataset (on Lim's page above).

---

## Notes on coverage & provenance

- List assembled by a parallel, web-grounded literature sweep (7 thematic search agents + 1 completeness critic, ~150 web searches). Every arXiv ID was verified by downloading and confirming the PDF title.
- The only confirmed-but-undownloaded item is the Ruiz-Meza 2022 TTDP review (ScienceDirect bot-block; it is genuinely open access — grab via your institutional access).
- `_download.sh` in this folder reproduces the downloads (with Wayback fallback for `munmund.net` and MDPI, which block direct curl).
