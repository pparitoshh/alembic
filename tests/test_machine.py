import json
import platform

from distillkit.machine import collect, markdown, parse_lscpu, parse_meminfo, parse_os_release

LSCPU = """CPU(s):                                  12
Model name:                              Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
Thread(s) per core:                      2
Core(s) per socket:                      6
Socket(s):                               1
CPU max MHz:                             4500.0000
L3 cache:                                12 MiB (1 instance)
Flags:                                   fpu sse sse2 avx avx2 fma"""


def test_parse_lscpu():
    c = parse_lscpu(LSCPU)
    assert (c["physical_cores"], c["logical_cpus"], c["max_mhz"], c["avx2"], c["avx512"]) == (6, 12, 4500.0, True, False)
    assert c["model"].startswith("Intel(R) Core(TM) i7-9750H")


def test_parse_meminfo_and_os():
    assert parse_meminfo("MemTotal:       16277808 kB\nSwapTotal:       4194300 kB\n") == {"ram_gb": 15.5, "swap_gb": 4.0}
    assert parse_os_release('NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04.4 LTS"\n') == "Ubuntu 24.04.4 LTS"


def test_collect_records_no_hostname():
    m = collect()
    assert platform.node() not in json.dumps(m)  # reports are committed to a public repo
    assert "| CPU |" in markdown(m)
