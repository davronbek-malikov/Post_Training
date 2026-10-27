"""Oracle "Out of capacity" bo'lsa: bepul ARM serverni (VM.Standard.A1.Flex) bo'sh joy chiqquncha qayta-qayta so'rash.

Bepul A1 serverlar tez-tez band bo'ladi. Skript har INTERVAL soniyada server yaratishga urinadi va joy chiqishi bilan
to'xtaydi. Faqat Always Free o'lchamlari ishlatiladi (A1.Flex, ko'pi bilan 4 OCPU / 24 GB) — pul yechilmaydi.

Tayyorgarlik (bir marta):
  pip install oci
  Oracle konsoli -> profil (o'ng yuqori) -> My profile -> API keys -> Add API key -> Generate API key pair
    -> Download private key -> Add -> ko'rsatilgan "Configuration file preview" matnini
    C:\\Users\\<siz>\\.oci\\config fayliga saqlang va undagi key_file= ni yuklab olingan .pem yo'liga o'zgartiring.
Ishga tushirish:
  python deploy/oracle/retry_create.py --ssh-key C:\\Users\\<siz>\\Downloads\\ssh-key-2026-10-27.key.pub
"""
import argparse
import time
from pathlib import Path

import oci

p = argparse.ArgumentParser()
p.add_argument("--ssh-key", required=True, help="konsoldan yuklab olingan public key (.pub)")
p.add_argument("--name", default="teachhelp")
p.add_argument("--ocpus", type=float, default=4)
p.add_argument("--memory", type=float, default=24)
p.add_argument("--interval", type=int, default=90, help="urinishlar orasidagi soniya")
args = p.parse_args()
assert args.ocpus <= 4 and args.memory <= 24, "Always Free limiti: 4 OCPU / 24 GB"

config = oci.config.from_file()
oci.config.validate_config(config)
tenancy = config["tenancy"]   # root compartment
compute = oci.core.ComputeClient(config)
network = oci.core.VirtualNetworkClient(config)
identity = oci.identity.IdentityClient(config)

ad = identity.list_availability_domains(tenancy).data[0].name
# Konsolda yaratilgan tarmoq (vcn-teachhelp / subnet-teachhelp) bo'lsa — o'sha; bo'lmasa avtomatik yaratiladi
subnets = network.list_subnets(tenancy).data
if subnets:
    subnet = next((s for s in subnets if "teachhelp" in s.display_name), subnets[0])
else:
    vcn = network.create_vcn(oci.core.models.CreateVcnDetails(
        compartment_id=tenancy, cidr_blocks=["10.0.0.0/16"], display_name="vcn-teachhelp")).data
    oci.wait_until(network, network.get_vcn(vcn.id), "lifecycle_state", "AVAILABLE")
    igw = network.create_internet_gateway(oci.core.models.CreateInternetGatewayDetails(
        compartment_id=tenancy, vcn_id=vcn.id, is_enabled=True, display_name="igw-teachhelp")).data
    network.update_route_table(vcn.default_route_table_id, oci.core.models.UpdateRouteTableDetails(route_rules=[
        oci.core.models.RouteRule(destination="0.0.0.0/0", network_entity_id=igw.id)]))
    subnet = network.create_subnet(oci.core.models.CreateSubnetDetails(
        compartment_id=tenancy, vcn_id=vcn.id, cidr_block="10.0.0.0/24", display_name="subnet-teachhelp")).data
    oci.wait_until(network, network.get_subnet(subnet.id), "lifecycle_state", "AVAILABLE")
image = compute.list_images(tenancy, operating_system="Canonical Ubuntu", operating_system_version="22.04",
                            shape="VM.Standard.A1.Flex", sort_by="TIMECREATED", sort_order="DESC").data[0]
print("AD:", ad, "| subnet:", subnet.display_name, "| image:", image.display_name)

details = oci.core.models.LaunchInstanceDetails(
    compartment_id=tenancy, availability_domain=ad, display_name=args.name, shape="VM.Standard.A1.Flex",
    shape_config=oci.core.models.LaunchInstanceShapeConfigDetails(ocpus=args.ocpus, memory_in_gbs=args.memory),
    source_details=oci.core.models.InstanceSourceViaImageDetails(image_id=image.id),
    create_vnic_details=oci.core.models.CreateVnicDetails(subnet_id=subnet.id, assign_public_ip=True),
    metadata={"ssh_authorized_keys": Path(args.ssh_key).read_text().strip()})

attempt = 0
while True:
    attempt += 1
    try:
        inst = compute.launch_instance(details).data
        break
    except oci.exceptions.ServiceError as e:
        busy = "capacity" in e.message.lower() or e.status in (429, 500)
        print(time.strftime("%H:%M:%S"), f"#{attempt}", "band, keyinroq" if busy else "XATO", "-", e.status, e.message[:120], flush=True)
        if not busy:
            raise
        time.sleep(args.interval * (2 if e.status == 429 else 1))

print("Yaratildi! Ishga tushishini kutyapman...")
oci.wait_until(compute, compute.get_instance(inst.id), "lifecycle_state", "RUNNING", max_wait_seconds=900)
vnic_id = compute.list_vnic_attachments(tenancy, instance_id=inst.id).data[0].vnic_id
print("TAYYOR. Public IP:", network.get_vnic(vnic_id).data.public_ip)
