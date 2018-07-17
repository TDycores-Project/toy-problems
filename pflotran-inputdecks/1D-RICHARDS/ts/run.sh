#!/bin/sh

display_help() {
    echo "Usage: $0 " >&2
    echo
    echo "   --pflotran  <path_to_pflotran_exe> Specify PFLTORAN exe"
    echo "   --ts_type   <beuler,cn>            Specify type of TS"
    echo "   -h, --help                         Display this message"
    echo
    exit 1
}

PFLOTRAN_EXE=
TS_TYPE=cn
while [ $# -gt 0 ]
do
  case "$1" in
    --pflotran ) PFLOTRAN_EXE="$2"; shift ;;
    --ts_type   ) TS_TYPE="$2";;
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

$PFLOTRAN_EXE -pflotranin t0001.in -flow_ts_type $TS_TYPE
$PFLOTRAN_EXE -pflotranin t0002.in -flow_ts_type $TS_TYPE
$PFLOTRAN_EXE -pflotranin t0004.in -flow_ts_type $TS_TYPE
$PFLOTRAN_EXE -pflotranin t0008.in -flow_ts_type $TS_TYPE
$PFLOTRAN_EXE -pflotranin t0016.in -flow_ts_type $TS_TYPE
$PFLOTRAN_EXE -pflotranin t0032.in -flow_ts_type $TS_TYPE
