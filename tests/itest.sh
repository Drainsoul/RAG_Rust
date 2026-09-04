#!/bin/zsh
HOST="0.0.0.0:3000"

## Test Document loading
DOCUMENT="Rust is a great programming language."
curl $HOST/doc -d '{"document":"'$DOCUMENT'"}'

## Test basic prompt
DOCUMENT="The outside world is scary I like staying indoors and with lots of blankets."
curl $HOST/doc -d '{"document":"'$DOCUMENT'"}'

DOCUMENT="I really like it when people say good thing about Rust."
curl $HOST/doc -d '{"document":"'$DOCUMENT'"}'

PROMPT="Rust"
curl $HOST/ -d '{"prompt":"'$PROMPT'"}'

PROMPT="Carbon"
curl $HOST/ -d '{"prompt":"'$PROMPT'"}'