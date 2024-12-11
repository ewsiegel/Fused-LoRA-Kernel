import pandas as pd
import numpy as np
# throughput numbers for 3090 in TFLOPs / s
THRUPUT = 35.5
THRUPUT_TENSR = 320
TENSR_i = 16
TENSR_j = 16
TENSR_k = 16
sizes = [
    # m, d, b, r
    # W mxd
    # x dxb
    # B mxr
    # A rxd
    (2048, 2048, 8, 8),
]

def calc_flops_sep(m, d, b, r):
    # returns num GIGAFLOPS when doing Ax then B(Ax)
    wx = m * b * 2*d
    ax = r * b * 2*d
    bax = m * b * 2*r
    return (wx + ax + bax) / 1000000000

def calc_flops_simul(m, d, b, r):
    # returns num GIGAFLOPS when doing BAx all at once
    wx = m * b * 2*d
    num_outputs = m * b
    bax = 2*r
    ax = r * 2*d # for every output, need to calc column in Ax (column has height r)
    return (wx + num_outputs * (ax + bax)) / 1000000000


def lower_bound_flops(gigaflops):
    # returns in ms
    return gigaflops / THRUPUT_TENSR

def num_bytes_sep(m, d, b, r):
    # return in GB
    #out = m * b
    #ax = r*d + d*b
    #bax = m*r + r*b
    #return 4 * (out + ax + bax) / 1000000000
    # return 4 * (m*d + 2*d*b + 5*m*b + r*d + 2*r*b + m*r) / 1000000000
    #ax = r*b * 2*d + r*b
    #bax = m*b*2*r + m*b
    #wx = m*b*2*d + m*b
    #return 4 * (ax+bax+wx) / 1000000000
    # return 4 * (r*b*(2*d+1) + m*b*(2*r + 2*d + 5)) / 1000000000

    # W_0x
    w0x = m*b*2*d + m*b # read, write
    # Ax
    ax = r*b*2*d + r*b
    #BAx
    bax = m*b*2*r + m*b
    # acc
    acc = 3*m*b # load each, write
    return 4 * (w0x + ax + bax + acc) / 1000000000





def num_bytes_simul(m, d, b, r):
    # return in GB
    #out = m * b
    #bax = 2*r
    #ax = r*d + d
    #return 4 * (out * (ax + bax)) / 1000000000
    # return 4 * (m*b*(2*r + r*d + d) + m*b) / 1000000000
    #return 4 * (m*d + d*b + m*b + m*b*r*d + m*r) / 1000000000
    #j_blocks = ceil(b / TENSR_j)
    #i_blocks = ceil(m / TENSR_i)
    #b = m*j_blocks*r
    #a = r*d*i_blocks*j_blocks
    #x = d*i_blocks*b
    #w = m*j_blocks*d 
    #write = m*b
    #return 4 * (b + a + x + w + write) / 1000000000
    # return 4 * (m*b*(d + r + r*d + d + 1)) / 1000000000

    # W_0x
    w0x = m*b*2*d

    #BAx
    bax = m*b*(r*d + 2*d + r)

    acc = m*b #only have to write
    return 4 * (w0x + bax + acc) / 1000000000


def lower_bound_DRAM(num_GB):
    #DRAM 936 GB/s throughput, return in ms
    return (num_GB / 936) * 1000



OUTER_ROWS = 4
OUTER_COLS = 32
INNER_ROWS = 16
INNER_COLS = 8

if __name__ == "__main__":
    # 1) Calc stats for 
    df = pd.DataFrame(sizes, columns=["size_m", "size_d", "size_b", "size_r"])
    df["GFLOPs Sep"] = calc_flops_sep(df["size_m"], df["size_d"], df["size_b"], df["size_r"])
    df["GFLOPs Simul"] = calc_flops_simul(df["size_m"], df["size_d"], df["size_b"], df["size_r"])
    df["Lower Bound FLOPs Sep (ms)"] = lower_bound_flops(df["GFLOPs Sep"])
    df["Lower Bound FLOPs Simul (ms)"] = lower_bound_flops(df["GFLOPs Simul"])
    df["GB Data Sep"] = num_bytes_sep(df["size_m"], df["size_d"], df["size_b"], df["size_r"])
    df["GB Data Simul"] = num_bytes_simul(df["size_m"], df["size_d"], df["size_b"], df["size_r"])
    df["Lower Bound DRAM Sep (ms)"] = lower_bound_DRAM(df["GB Data Sep"])
    df["Lower Bound DRAM Simul (ms)"] = lower_bound_DRAM(df["GB Data Simul"])
    #df["Max TFLOPs / s"] = df["GigaFLOPs"] / np.maximum(df["Lower Bound FLOPs (ms)"], df["Lower Bound DRAM (ms)"])
    #df["Num ThreadBlocks"] = (df["size_i"] / (OUTER_ROWS * INNER_ROWS)) * (df["size_j"] / (OUTER_COLS * INNER_COLS))
    print(df)
