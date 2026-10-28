#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "live_ppo_infer.h"

int main(int argc, char **argv) {
    if(argc != 3) return 2;
    FILE *p = fopen(argv[1], "rb"), *v = fopen(argv[2], "rb");
    if(!p || !v) return 3;
    uint8_t *data = malloc(52 + 346184), header[52];
    float x[LIVE_PPO_NI], logits[LIVE_PPO_NA];
    if(!data || fread(header, 1, sizeof(header), p) != sizeof(header) ||
       fread(data + 52, 1, 346184, p) != 346184 || fread(x, sizeof(float), LIVE_PPO_NI, v) != LIVE_PPO_NI) return 4;
    memcpy(data, header, sizeof(header));
    live_ppo_infer_logits(data, x, logits);
    putchar('[');
    for(int i = 0; i < LIVE_PPO_NA; ++i) printf(i ? ",%.9g" : "%.9g", logits[i]);
    puts("]");
    free(data); fclose(p); fclose(v);
    return 0;
}
