#!/bin/sh
set -eu

OUT=${1:-/certs}
mkdir -p "$OUT"
rm -f "$OUT"/*.pem "$OUT"/*.srl

openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "$OUT/ca-key.pem" \
  -out "$OUT/ca.pem" \
  -days 2 \
  -subj "/CN=VertiMosaic Demo CA"

issue_server() {
  name="$1"
  openssl req -newkey rsa:2048 -nodes \
    -keyout "$OUT/${name}-key.pem" \
    -out "$OUT/${name}.csr" \
    -subj "/CN=${name}"
  printf 'subjectAltName=DNS:%s\nextendedKeyUsage=serverAuth\n' "$name" > "$OUT/${name}.ext"
  openssl x509 -req \
    -in "$OUT/${name}.csr" \
    -CA "$OUT/ca.pem" \
    -CAkey "$OUT/ca-key.pem" \
    -CAcreateserial \
    -out "$OUT/${name}.pem" \
    -days 2 \
    -extfile "$OUT/${name}.ext"
  rm -f "$OUT/${name}.csr" "$OUT/${name}.ext"
}

for name in telecom insurance retail; do
  issue_server "$name"
done

openssl req -newkey rsa:2048 -nodes \
  -keyout "$OUT/bank-client-key.pem" \
  -out "$OUT/bank-client.csr" \
  -subj "/CN=bank"
printf 'extendedKeyUsage=clientAuth\n' > "$OUT/bank-client.ext"
openssl x509 -req \
  -in "$OUT/bank-client.csr" \
  -CA "$OUT/ca.pem" \
  -CAkey "$OUT/ca-key.pem" \
  -CAcreateserial \
  -out "$OUT/bank-client.pem" \
  -days 2 \
  -extfile "$OUT/bank-client.ext"
rm -f "$OUT/bank-client.csr" "$OUT/bank-client.ext" "$OUT/ca-key.pem" "$OUT/ca.srl"
chmod 600 "$OUT"/*-key.pem

echo "Generated short-lived demo CA, server certificates, and bank client certificate in $OUT"
