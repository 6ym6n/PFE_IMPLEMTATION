#!/usr/bin/env bash
# Download open-access POI-itinerary papers. Validates each file is a real PDF.
# Usage: bash articles/_download.sh
set -u
cd "$(dirname "$0")"

UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# url|||filename   (one per line; dedup already applied)
read -r -d '' LIST <<'EOF'
https://people.sutd.edu.sg/~kwanhui_lim/publications/2018-KAIS-personalTour.pdf|||2015-2018_Lim_PersTour_KAIS-extended.pdf
https://web.archive.org/web/2id_/https://www.munmund.net/pubs/www_10_2.pdf|||2010_DeChoudhury_SocialBreadcrumbs_HT-WWW.pdf
https://lirias.kuleuven.be/retrieve/91f75023-e620-4f54-90c2-5ada4495c62c/|||2009_Souffriau_ILS-TOPTW_COR.pdf
https://arxiv.org/pdf/1608.07051|||2016_Chen_LearningPointsRoutes_CIKM.pdf
https://www.microsoft.com/en-us/research/wp-content/uploads/2010/01/24-2010-acmmm-photo2trip-mmfat06245-Xin.pdf|||2010_Lu_Photo2Trip_ACMMM.pdf
https://kwanhui.github.io/publications/2019-KAIS-surveyTourRec.pdf|||2019_Lim_TourRecTripPlanning-Survey_KAIS.pdf
https://kwanhui.github.io/publications/2024-ASOC-surveyDeepTour.pdf|||2024_Halder_ItineraryRec-Survey_ASOC.pdf
https://arxiv.org/pdf/2410.02191|||2024_Zhang_POIRec-Survey.pdf
https://arxiv.org/pdf/1706.09067|||2017_Chen_StructuredRecommendation.pdf
https://arxiv.org/pdf/1708.05165|||2017_Menon_RevisitingRevisits-TrajRec.pdf
https://kwanhui.github.io/publications/2016-UMAP-tourRecDC.pdf|||2016_Lim_PersonalizedItineraries_UMAP.pdf
https://kwanhui.github.io/publications/2018-LocWeb-tourRecMustVisit.pdf|||2018_Taylor_TourMustSee_LocWeb.pdf
https://www.ijcai.org/proceedings/2017/0255.pdf|||2017_He_CategoryAware-ListwiseBPR_IJCAI.pdf
http://www.shuwu.name/sw/Distance2Pre.pdf|||2019_Cui_Distance2Pre_PAKDD.pdf
https://arxiv.org/pdf/1808.08023|||2018_He_JointContextAwareEmbedding-Trip_ICDE.pdf
http://cake.fiu.edu/Publications/Gao+al-19-DA.DeepTrip_Adversarially_Understanding_Human_Mobility.ACM.downloaded.pdf|||2019_Gao_DeepTrip_SIGSPATIAL.pdf
https://arxiv.org/pdf/1907.08489|||2019_Wang_NeuralAstar-RouteRec_KDD.pdf
https://arxiv.org/pdf/2109.11731|||2021_Jiang_AdversarialNeuralTrip.pdf
https://arxiv.org/pdf/2109.00968|||2022_Gao_SelfTrip_KBS.pdf
https://arxiv.org/pdf/2109.03535|||2021_Rashid_DeepAltTrip_TKDE.pdf
https://web.archive.org/web/2id_/https://www.mdpi.com/2220-9964/8/10/433/pdf|||2019_Liao_DRPS-POISequence_IJGI.pdf
https://arxiv.org/pdf/2103.02464|||2021_Ho_TourRec-POIEmbedding_IUI.pdf
https://arxiv.org/pdf/2212.13900|||2022_Ho_POIBERT_BigData.pdf
https://arxiv.org/pdf/1506.03134|||2015_Vinyals_PointerNetworks_NeurIPS.pdf
https://arxiv.org/pdf/2310.19886|||2023_Ho_BTRec_RecTour.pdf
https://arxiv.org/pdf/2311.11071|||2023_Ho_SBTRec_BigData.pdf
https://arxiv.org/pdf/2311.12355|||2023_Ho_LanguageModels-TourItinerary_IJCAI-PMAI.pdf
https://arxiv.org/pdf/2303.04741|||2022_Yang_GETNext_SIGIR.pdf
https://arxiv.org/pdf/2507.19798|||2024_Shu_ARTrip-Repetitions_SIGIR.pdf
https://arxiv.org/pdf/2402.01622|||2024_Xie_TravelPlanner_ICML.pdf
https://aclanthology.org/2024.emnlp-industry.104.pdf|||2024_Tang_ItiNera_EMNLP.pdf
https://arxiv.org/pdf/2502.20508|||2025_Chaudhuri_TripCraft_ACL.pdf
https://aclanthology.org/2025.findings-acl.503.pdf|||2025_Shen_TripTailor_ACL-Findings.pdf
https://arxiv.org/pdf/2412.13682|||2024_Shao_ChinaTravel.pdf
https://aclanthology.org/2024.emnlp-demo.25.pdf|||2024_Ju_ToTheGlobe-TTG_EMNLP-Demo.pdf
https://arxiv.org/pdf/2404.11891|||2025_Hao_LLM-FormalVerification_NAACL.pdf
https://arxiv.org/pdf/2409.08069|||2024_Chen_TravelAgent.pdf
https://arxiv.org/pdf/2505.10922|||2025_Wang_Vaiage-MultiAgent.pdf
https://arxiv.org/pdf/2510.21329|||2025_TripTide_AdaptivePlanning.pdf
https://arxiv.org/pdf/2410.17333|||2024_Ren_LLMsReady-TravelPlanning.pdf
https://kwanhui.github.io/publications/2025-TORS-deepTourRec.pdf|||2025_Halder_DLIR-DynamicPOI_TORS.pdf
https://arxiv.org/pdf/2502.17345|||2025_PlusTour-SmartTourism_ComputerNetworks.pdf
https://arxiv.org/pdf/2504.08694|||2025_Ni_TP-RAG_EMNLP.pdf
https://arxiv.org/pdf/1803.08475|||2019_Kool_AttentionRoutingProblems_ICLR.pdf
https://arxiv.org/pdf/2505.11239|||2025_Wongso_Massive-STEPS_Dataset.pdf
https://arxiv.org/pdf/2605.03308|||2026_Revisiting-LLM-TravelPlanning.pdf
EOF
# NOTE: 2022 Ruiz-Meza TTDP review (Operations Research Perspectives, open access) is
# bot-blocked by ScienceDirect; download manually via institutional access:
# https://doi.org/10.1016/j.orp.2022.100228

ok=0; fail=0
echo "==== Downloading $(echo "$LIST" | grep -c '|||') papers ===="
while IFS= read -r line; do
  [ -z "$line" ] && continue
  url="${line%%|||*}"
  fn="${line##*|||}"
  # download
  curl -sSL --max-time 90 --retry 2 -A "$UA" -o "$fn" "$url" 2>/dev/null
  # validate: exists, >20KB, starts with %PDF
  if [ -f "$fn" ]; then
    magic=$(head -c 4 "$fn" 2>/dev/null)
    size=$(wc -c < "$fn" 2>/dev/null | tr -d ' ')
    if [ "$magic" = "%PDF" ] && [ "${size:-0}" -gt 20000 ]; then
      printf 'OK    %8s  %s\n' "$size" "$fn"
      ok=$((ok+1))
    else
      printf 'FAIL  (not pdf / too small: %s b)  %s  <- %s\n' "${size:-0}" "$fn" "$url"
      rm -f "$fn"
      fail=$((fail+1))
    fi
  else
    printf 'FAIL  (no file)  %s  <- %s\n' "$fn" "$url"
    fail=$((fail+1))
  fi
done <<< "$LIST"

echo "==== done: $ok ok, $fail failed ===="
