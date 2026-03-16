#!/bin/bash
# โหลดผลทั้งหมดจาก Vast.ai แล้วค่อย destroy
# Usage: bash scripts/download_all_vastai.sh

set -e
BASE="/Users/pop7/Code/Lanta/results"
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no"

echo "🔍 Checking V1 (39591)..."
V1_STATUS=$(ssh $SSH_OPTS -p 39591 root@ssh3.vast.ai 'supervisorctl status' 2>/dev/null || echo "OFFLINE")
echo "$V1_STATUS"

echo ""
echo "🔍 Checking V2 (10481)..."
V2_STATUS=$(ssh $SSH_OPTS -p 10481 root@ssh3.vast.ai 'supervisorctl status' 2>/dev/null || echo "OFFLINE")
echo "$V2_STATUS"

# Check if any jobs still running
if echo "$V1_STATUS" | grep -q "RUNNING"; then
    echo ""
    echo "⚠️  V1 ยังมี job running อยู่! โหลดเฉพาะที่เสร็จแล้ว"
fi
if echo "$V2_STATUS" | grep -q "RUNNING"; then
    echo ""
    echo "⚠️  V2 ยังมี job running อยู่! โหลดเฉพาะที่เสร็จแล้ว"
fi

echo ""
echo "📥 Downloading from V1..."
mkdir -p "$BASE/vastai-remaining" "$BASE/vastai-orpo" "$BASE/vastai-v1-p18p15"

scp $SSH_OPTS -P 39591 "root@ssh3.vast.ai:/workspace/results/remaining/*" "$BASE/vastai-remaining/" 2>/dev/null && echo "  ✅ remaining" || echo "  ⏳ remaining not ready"
scp $SSH_OPTS -P 39591 "root@ssh3.vast.ai:/workspace/results/orpo/*" "$BASE/vastai-orpo/" 2>/dev/null && echo "  ✅ orpo" || echo "  ⏳ orpo not ready"
scp $SSH_OPTS -P 39591 "root@ssh3.vast.ai:/workspace/results/p18_p15/*" "$BASE/vastai-v1-p18p15/" 2>/dev/null && echo "  ✅ p18_p15" || echo "  ⏳ p18_p15 not ready"

echo ""
echo "📥 Downloading from V2..."
mkdir -p "$BASE/vastai-v2-final" "$BASE/vastai-p6-9b"

scp $SSH_OPTS -P 10481 "root@ssh3.vast.ai:/workspace/results/v2_extra/*" "$BASE/vastai-v2-final/" 2>/dev/null && echo "  ✅ v2_extra" || echo "  ⏳ v2_extra not ready"
scp $SSH_OPTS -P 10481 "root@ssh3.vast.ai:/workspace/results/p6_9b/*" "$BASE/vastai-p6-9b/" 2>/dev/null && echo "  ✅ p6_9b" || echo "  ⏳ p6_9b not ready"

echo ""
echo "📊 Downloaded files:"
echo "  remaining: $(ls $BASE/vastai-remaining/*.json 2>/dev/null | wc -l) files"
echo "  orpo:      $(ls $BASE/vastai-orpo/*.json 2>/dev/null | wc -l) files"
echo "  p18_p15:   $(ls $BASE/vastai-v1-p18p15/*.json 2>/dev/null | wc -l) files"
echo "  v2_extra:  $(ls $BASE/vastai-v2-final/*.json 2>/dev/null | wc -l) files"
echo "  p6_9b:     $(ls $BASE/vastai-p6-9b/*.json 2>/dev/null | wc -l) files"

echo ""
echo "✅ โหลดเสร็จ! ไป destroy ได้ที่: https://cloud.vast.ai/instances/"
