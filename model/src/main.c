#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum ERROR {
    PASS,
    FAIL
};

int
concatenate_strings(char *result, char *strings[], int count) {
    for (int idx = 0; idx < count; idx++) {
        strcat(result, strings[idx]);
    }

    return PASS;
}

int
run_model(char *filename) {
    FILE *fp;
    char *command = malloc(2 + strlen(filename) + strlen("python3"));

    char *command_fields[] = {"python3", " ", filename};
    if (concatenate_strings(command, command_fields, 3) == 1) return FAIL;

    fp = popen(command, "r");
    if (!fp) return FAIL;
    pclose(fp);
    free(command);

    return PASS;
}

int
main(void) {
    enum ERROR error;

    error = run_model("test.py");

    return error;
}