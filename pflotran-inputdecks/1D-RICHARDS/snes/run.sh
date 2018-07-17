#!/bin/sh

display_help() {
    echo "Usage: $0 " >&2
    echo
    echo "   --pflotran  <path_to_pflotran_exe> Specify PFLTORAN exe"
    echo "   -h, --help                         Display this message"
    echo
    exit 1
}

PFLOTRAN_EXE=
while [ $# -gt 0 ]
do
  case "$1" in
    --pflotran ) PFLOTRAN_EXE="$2"; shift ;;
    -*)
      display_help
      exit 0
      ;;
    -h | --help)
      display_help
      exit 0
      ;;
    *)  break;;	# terminate while loop
  esac
  shift
done

if [ -z "$PFLOTRAN_EXE" ]
then
  echo "\$PFLOTRAN_EXE is unspecified"
  display_help
fi

$PFLOTRAN_EXE -pflotranin t0001.in
$PFLOTRAN_EXE -pflotranin t0002.in
$PFLOTRAN_EXE -pflotranin t0004.in
$PFLOTRAN_EXE -pflotranin t0008.in
$PFLOTRAN_EXE -pflotranin t0016.in
$PFLOTRAN_EXE -pflotranin t0032.in
