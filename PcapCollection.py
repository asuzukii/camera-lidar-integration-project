import subprocess


def externalWiresharkCollection(save_path: str, networkInterface: str, collect_duration: int):
  tshark = "C:/Program Files/Wireshark/tshark.exe";
  command = [tshark, "-i", str(networkInterface), "-w", save_path, "-a", "duration:{}".format(collect_duration), "-F", "pcap"]

  rc = subprocess.run(command)

if __name__ == "__main__":
  save_path = r"C:\Users\TianyuZhao.LAPTOP-UNOKG67S\workspace\projects\sdk_test\py_sdk.pcap"
  capture_time = 10
  externalWiresharkCollection(save_path, "Ethernet 2", capture_time)