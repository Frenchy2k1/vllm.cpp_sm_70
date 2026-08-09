#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DESTINATION" >&2
  exit 2
fi

destination=$1
archive_name=sde-external-10.13.1-2026-07-28-lin.tar.xz
archive_sha256=94e97d623fec54385686e1e7ba65ebc9941748c05ee451423948334892bf2b50
archive_url=https://downloadmirror.intel.com/924984/$archive_name

if [[ -e "$destination" ]]; then
  echo "Intel SDE destination already exists: $destination" >&2
  exit 1
fi

mkdir -p "$destination"
archive="$destination/$archive_name"
curl --fail --location --retry 3 --output "$archive" "$archive_url"
printf '%s  %s\n' "$archive_sha256" "$archive" | sha256sum --check
tar --extract --xz --file "$archive" --directory "$destination" --strip-components=1
rm -f -- "$archive"
test -x "$destination/sde64"
