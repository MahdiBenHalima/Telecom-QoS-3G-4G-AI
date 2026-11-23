def generate_explanation_4g(rsrp_class, rsrq_class, throughput_class, decision, zone=None):
    zone_text = f"dans la zone {zone}" if zone else "dans cette zone"
    return (f"En 4G, la couverture radio {zone_text} est {rsrp_class.lower()}, "
            f"la qualité radio est {rsrq_class.lower()} et le débit descendant est "
            f"{throughput_class.lower()}. Conclusion : {decision}.")

def generate_explanation_3g(rscp_class, ecno_class, decision, zone=None):
    zone_text = f"dans la zone {zone}" if zone else "dans cette zone"
    return (f"En 3G, la puissance du signal {zone_text} est {rscp_class.lower()} "
            f"et la qualité Ec/N0 est {ecno_class.lower()}. Conclusion : {decision}.")
