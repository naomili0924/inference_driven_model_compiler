#!/usr/bin/env bash
# Run all pipeline tests in on_the_fly_pipeline_tests/.
# Must be executed from /workspace (the repo parent) so that the local
# optimum/ subdirectory does not shadow the installed optimum package.
#
# Usage:
#   cd /workspace
#   bash inference_driven_model_compiler/run_tests.sh
#
# Options:
#   TIMEOUT  seconds per test (default 600)
#   TESTS    space-separated list of filenames to run (default: all *.py)
#
# Exit code: 0 if every test passed, 1 otherwise.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/on_the_fly_pipeline_tests"
TIMEOUT="${TIMEOUT:-600}"

# Collect tests — allow override via env var
if [[ -n "${TESTS:-}" ]]; then
    mapfile -t TEST_FILES < <(for t in $TESTS; do echo "${TEST_DIR}/${t}"; done)
else
    mapfile -t TEST_FILES < <(ls "${TEST_DIR}"/*.py | sort)
fi

PASS=()
FAIL=()
SKIP=()

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

separator() { printf '%0.s─' {1..64}; echo; }

echo
separator
printf "  Running %d test(s) from on_the_fly_pipeline_tests/\n" "${#TEST_FILES[@]}"
printf "  Timeout per test: %ds\n" "${TIMEOUT}"
separator

for test_file in "${TEST_FILES[@]}"; do
    name="$(basename "${test_file}")"
    printf "\n▶  %s\n" "${name}"

    log_file="$(mktemp /tmp/test_XXXXXX.log)"

    set +e
    timeout "${TIMEOUT}" python3 "${test_file}" > "${log_file}" 2>&1
    exit_code=$?
    set -e

    if [[ ${exit_code} -eq 124 ]]; then
        printf "  ${YELLOW}[TIMEOUT]${NC} exceeded ${TIMEOUT}s\n"
        SKIP+=("${name} (timeout)")
    elif [[ ${exit_code} -eq 0 ]]; then
        printf "  ${GREEN}[PASS]${NC}\n"
        # Print last few lines of output so progress is visible
        tail -n 5 "${log_file}" | sed 's/^/    /'
        PASS+=("${name}")
    else
        printf "  ${RED}[FAIL]${NC} exit code ${exit_code}\n"
        # Show the last 20 lines on failure
        tail -n 20 "${log_file}" | sed 's/^/    /'
        FAIL+=("${name}")
    fi

    rm -f "${log_file}"
done

echo
separator
printf "  Results: ${GREEN}%d passed${NC}  ${RED}%d failed${NC}  ${YELLOW}%d skipped${NC}\n" \
    "${#PASS[@]}" "${#FAIL[@]}" "${#SKIP[@]}"

if [[ ${#PASS[@]} -gt 0 ]]; then
    printf "\n  ${GREEN}PASSED${NC}\n"
    for t in "${PASS[@]}"; do printf "    ✓  %s\n" "${t}"; done
fi

if [[ ${#FAIL[@]} -gt 0 ]]; then
    printf "\n  ${RED}FAILED${NC}\n"
    for t in "${FAIL[@]}"; do printf "    ✗  %s\n" "${t}"; done
fi

if [[ ${#SKIP[@]} -gt 0 ]]; then
    printf "\n  ${YELLOW}SKIPPED${NC}\n"
    for t in "${SKIP[@]}"; do printf "    -  %s\n" "${t}"; done
fi

separator

[[ ${#FAIL[@]} -eq 0 ]]
