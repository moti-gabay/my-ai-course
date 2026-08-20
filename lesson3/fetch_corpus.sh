#!/usr/bin/env bash
# Re-fetch the insurance corpus (Assignment 3).
# Six PDF policy documents from five companies plus one federal form, and the
# NFIP Standard Flood Insurance Policy pulled from eCFR as Markdown.
set -u

UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
D="$(cd "$(dirname "$0")" && pwd)/corpus"
mkdir -p "$D"

dl() {
  if curl -sL --fail --max-time 180 -A "$UA" -o "$D/$1" "$2"; then
    echo "ok    $1  $(stat -c%s "$D/$1") bytes"
  else
    echo "FAIL  $1"
  fi
}

# --- Auto (Allstate) ---
dl allstate_auto_policy.pdf \
  'https://delivery.contenthub.allstate.com/api/public/content/edf6f4419f6b4d7787b736308bdcb612?v=94d767d3'

# --- Pet (Embrace) ---
dl embrace_pet_policy.pdf \
  'https://assets.ctfassets.net/nx3pzsky0bc9/7Dj04o1HSmbAPpDBiPvwts/01301775df64181efc84478a2b450338/PET50-202301.pdf'

# --- Pet (Nationwide) ---
dl nationwide_pet_medical_plan.pdf \
  'https://www.petinsurance.com/images/VSSimages/media/pdf/VS-G-2(2-17)-NCC-Medical-Plan-SAMPLE.pdf'

# --- Health (MHBP / Aetna, FEHB Evidence of Coverage) ---
dl mhbp_health_evidence_of_coverage.pdf \
  'https://mhbppostal.com/wp-content/uploads/2024/02/2024-Evidence-of-Coverage.pdf'

# --- Travel (Allianz, US / Florida certificate) ---
dl allianz_travel_basic_fl.pdf \
  'https://www.travelinsurance.com/brochure/Allianz/Allianz_Basic_FL_0216.pdf'

# --- Travel (Allianz, UK policy wording) ---
dl allianz_travel_policy_wording.pdf \
  'https://www.bigcattravelinsurance.com/policy-pdf/Policy-Wording-2020-Allianz-14March20.pdf'

# --- Flood (NFIP Standard Flood Insurance Policy, 44 CFR Part 61) ---
# Fetched as HTML from eCFR and converted to Markdown by html_to_md.py so the
# corpus contains a second file format.
curl -sL --fail --max-time 180 -A "$UA" \
  -o /tmp/ecfr_44cfr61.html \
  'https://www.ecfr.gov/api/renderer/v1/content/enhanced/current/title-44?part=61' \
  && python3 "$(dirname "$0")/html_to_md.py" /tmp/ecfr_44cfr61.html "$D/nfip_flood_policy_44cfr61.md" \
  && echo "ok    nfip_flood_policy_44cfr61.md  $(stat -c%s "$D/nfip_flood_policy_44cfr61.md") bytes"

ls -la "$D"
