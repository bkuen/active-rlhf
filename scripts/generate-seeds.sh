#!/usr/bin/env sh
# generate_seeds.sh — produce 10 random 32‑bit signed integer seeds (1..2147483647)

COUNT=10   # how many seeds to generate

for i in $(seq 1 $COUNT); do
  # 1) grab 4 bytes → unsigned 32‑bit integer (0…2^32-1)
  RAND=$(od -An -N4 -tu4 < /dev/urandom | tr -d ' ')

  # 2) mask to 31 bits → value in 0…2^31-1
  SEED=$(( RAND & 0x7fffffff ))

  # 3) if we got 0, bump up to 1 so seed ∈ [1, 2147483647]
  if [ "$SEED" -eq 0 ]; then
    SEED=1
  fi

  echo "$SEED"
done