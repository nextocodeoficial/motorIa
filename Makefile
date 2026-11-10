CC      ?= clang
CFLAGS  ?= -std=c99 -O3 -Wall -Wextra -mcpu=apple-m4 -fblocks
CFLAGS  += -Iinclude -D_DARWIN_C_SOURCE
LDLIBS  := -lm -lpthread

BUILD   := build
LIB_SRC := $(wildcard src/tensor/*.c src/model/*.c src/storage/*.c src/tokenizer/*.c)
LIB_OBJ := $(LIB_SRC:%.c=$(BUILD)/%.o)
TESTS   := $(patsubst tests/%.c,$(BUILD)/tests/%,$(wildcard tests/test_*.c))

.PHONY: all test clean e2e
all: $(BUILD)/mia $(BUILD)/mia-dump $(BUILD)/mia-serve

$(BUILD)/%.o: %.c $(wildcard include/mia/*.h)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/libmia.a: $(LIB_OBJ)
	ar rcs $@ $^

$(BUILD)/mia: src/app/main.c $(BUILD)/libmia.a
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/mia-serve: src/app/serve.c $(BUILD)/libmia.a
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/mia-dump: src/app/dump.c $(BUILD)/libmia.a
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/tests/%: tests/%.c tests/check.h $(BUILD)/libmia.a
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< $(BUILD)/libmia.a -o $@ $(LDLIBS)

test: $(TESTS)
	@set -e; for t in $(TESTS); do echo "== $$t"; $$t; done; echo "ALL C TESTS PASSED"

e2e: $(BUILD)/mia-dump $(BUILD)/mia $(BUILD)/mia-serve
	cd tools && MIA_DUMP=$(abspath $(BUILD)/mia-dump) MIA_BIN=$(abspath $(BUILD)/mia) MIA_SERVE=$(abspath $(BUILD)/mia-serve) go test -count=1 ./...

clean:
	rm -rf $(BUILD)

# ---- chat web UI -------------------------------------------------------
.PHONY: web chat
web:
	cd web && npm run build

$(BUILD)/mia-web: web $(BUILD)/mia-serve
	cd tools && go build -o ../$(BUILD)/mia-web ./cmd/mia-web

MODEL ?= /Volumes/nuevo/models/qwen3-30b-a3b-q8.mia
RAM   ?= 6G
chat: $(BUILD)/mia-web
	$(BUILD)/mia-web -model $(MODEL) -ram $(RAM)
