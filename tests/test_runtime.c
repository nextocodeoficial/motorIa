#include <stdlib.h>
#include <string.h>

#include "check.h"
#include "mia/runtime.h"

static void test_first_turn_without_thinking(void) {
    char *s = mia_chat_turn("hola", 0, 0);
    CHECK(strcmp(s, "<|im_start|>user\nhola<|im_end|>\n<|im_start|>assistant\n"
                    "<think>\n\n</think>\n\n") == 0);
    free(s);
}

static void test_thinking_leaves_think_block_open(void) {
    char *s = mia_chat_turn("hola", 1, 0);
    CHECK(strcmp(s, "<|im_start|>user\nhola<|im_end|>\n<|im_start|>assistant\n") ==
          0);
    free(s);
}

static void test_continuing_closes_previous_reply(void) {
    char *s = mia_chat_turn("y?", 0, 1);
    CHECK(strncmp(s, "<|im_end|>\n<|im_start|>user\ny?", 30) == 0);
    free(s);
}

static void test_open_rejects_missing_file(void) {
    mia_runtime rt;
    char err[256] = {0};
    CHECK(mia_runtime_open(&rt, "/nonexistent.mia", 1e9, 128, err,
                           sizeof(err)) != 0);
    CHECK(strstr(err, "nonexistent") != NULL);
}

int main(void) {
    RUN(test_first_turn_without_thinking);
    RUN(test_thinking_leaves_think_block_open);
    RUN(test_continuing_closes_previous_reply);
    RUN(test_open_rejects_missing_file);
    return CHECK_SUMMARY();
}
