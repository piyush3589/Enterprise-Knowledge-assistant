"""
Knowledge base seeder — ingests IoT/Industrial equipment documents into ChromaDB.
Run once: python data/seed_knowledge.py

Documents cover: equipment specs, maintenance procedures, safety protocols,
troubleshooting guides, and compliance standards — typical Faclon domain content.
"""

"""
Seed script for Enterprise Knowledge Assistant.
Chunks documents before storing — improves retrieval precision.
"""

import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_PATH = "./chroma_db"

DOCUMENTS = [
    {
        "id": "doc_boiler_001",
        "content": """
EQUIPMENT: Industrial Boiler B-01
MODEL: Thermax TB-500
LOCATION: Plant A, Zone 2
INSTALLATION DATE: March 2021

SPECIFICATIONS:
- Rated capacity: 500 kg/hr steam output
- Operating pressure: 8.5 bar (max 10 bar)
- Operating temperature: 60-85°C (critical threshold: 95°C)
- Fuel type: Natural gas (LPG backup)
- Heat exchanger type: Shell and tube, 4-pass

MAINTENANCE SCHEDULE:
- Daily: Check water level, pressure gauges, burner operation
- Weekly: Blow down procedure, safety valve test
- Monthly: Clean heat exchanger fins, inspect flue gas path
- Annual: Full inspection by certified engineer, pressure vessel test

CRITICAL COMPONENTS:
- Control valve CV-201: Regulates steam output, inspect every 6 months
- Safety relief valve SRV-101: Must not exceed 9.5 bar, replace every 2 years
- Water level controller WLC-01: Calibrate quarterly
- Burner management system BMS-01: Software update annually

SHUTDOWN PROCEDURE:
1. Reduce burner firing rate to minimum
2. Close main steam valve MSV-01
3. Allow pressure to reduce naturally - do not vent rapidly
4. When pressure < 1 bar, open drain valve DV-201
5. Log shutdown in maintenance register
""",
        "metadata": {"category": "equipment_spec", "equipment": "boiler", "location": "Plant A"},
    },
    {
        "id": "doc_boiler_maintenance_001",
        "content": """
MAINTENANCE PROCEDURE: Boiler Heat Exchanger Cleaning
DOCUMENT: MP-BOILER-003
REVISION: 4
APPLICABLE EQUIPMENT: Thermax TB-500, Plant A Zone 2

FREQUENCY: Every 3 months or when pressure drop across heat exchanger exceeds 0.5 bar

TOOLS REQUIRED:
- High pressure water jet (min 150 bar)
- Fin comb set
- Inspection camera
- Personal protective equipment (heat resistant gloves, goggles, respirator)

PROCEDURE:
1. Isolate boiler from steam distribution - close MSV-01 and lock out
2. Allow boiler to cool to below 40°C (minimum 4 hours)
3. Open inspection ports IP-01 through IP-04
4. Insert inspection camera - document condition before cleaning
5. Apply chemical descaler solution - leave 30 minutes
6. High pressure water jet cleaning - work systematically tube by tube
7. Inspect tubes for corrosion, pitting, or blockage
8. Replace any tubes showing wall thickness below 3mm
9. Reassemble inspection ports - torque to 45 Nm
10. Perform hydraulic pressure test at 1.5x operating pressure
11. Document findings and sign off in maintenance register

ACCEPTANCE CRITERIA:
- No visible scale buildup
- All tubes free-flowing (verify with flow test)
- No cracks or corrosion on tube sheet
- Pressure test holds for minimum 30 minutes without drop

SAFETY WARNING: Never open inspection ports while boiler is pressurised.
""",
        "metadata": {"category": "maintenance_procedure", "equipment": "boiler", "location": "Plant A"},
    },
    {
        "id": "doc_pump_001",
        "content": """
EQUIPMENT: Centrifugal Pump P-03
MODEL: Grundfos CR 45-3
LOCATION: Plant B, Zone 1
PURPOSE: Coolant circulation for boiler cooling loop

SPECIFICATIONS:
- Flow rate: 45-75 L/min (design point: 60 L/min)
- Head: 32 metres
- Motor: 7.5 kW, 415V, 3-phase
- Shaft seal type: Mechanical seal, silicon carbide faces
- Bearing type: Deep groove ball bearings, grease lubricated
- Vibration limit: 0.5-4.0 mm/s RMS (alarm: 6.0, trip: 8.0)
- NPSH required: 3.2 metres

INSTRUMENTATION:
- Flow meter FT-201: Ultrasonic, 4-20mA output
- Vibration sensor VS-301: Piezoelectric, mounted on bearing housing
- Discharge pressure PT-201: 0-10 bar, 4-20mA
- Motor temperature TE-401: PT100, alarm at 80°C

BEARING REPLACEMENT SCHEDULE:
- Grease lubrication: Every 2000 hours operation
- Full bearing replacement: Every 18 months or 15000 hours
- Check shaft alignment: Every 6 months and after any maintenance

COMMON FAILURE MODES:
1. Cavitation: Noise like gravel, low flow, high vibration - check NPSH, inlet pressure
2. Bearing failure: High vibration, noise, heat - check lubrication, alignment
3. Mechanical seal leak: Visible leak at shaft - replace seal immediately
4. Impeller wear: Reduced flow at same power - measure impeller clearance
""",
        "metadata": {"category": "equipment_spec", "equipment": "pump", "location": "Plant B"},
    },
    {
        "id": "doc_pump_maintenance_001",
        "content": """
MAINTENANCE PROCEDURE: Centrifugal Pump Bearing Replacement
DOCUMENT: MP-PUMP-007
REVISION: 2
APPLICABLE EQUIPMENT: Grundfos CR series, Plant B

TRIGGER CONDITIONS:
- Vibration exceeds 6.0 mm/s RMS for more than 15 minutes
- Bearing temperature exceeds 75°C
- Scheduled replacement at 18-month interval
- Unusual noise during operation (grinding, screeching)

PARTS REQUIRED:
- Bearing set (drive end + non-drive end): Part# GF-CR45-BRG-SET
- Mechanical seal: Part# GF-CR45-SEAL-001
- Coupling insert: Part# GF-CR45-COUP-001
- O-ring kit: Part# GF-CR45-ORING-KIT

PROCEDURE:
1. LOTO - isolate motor electrically, lock out MCC panel
2. Close isolation valves IV-301 (inlet) and IV-302 (outlet)
3. Drain pump casing via drain plug DP-01
4. Disconnect coupling - mark alignment position first
5. Remove motor and set aside on clean surface
6. Extract pump shaft assembly
7. Remove old bearings using bearing puller - do not hammer
8. Clean bearing housing with lint-free cloth and solvent
9. Install new bearings - use bearing heater, never hammer
10. Install new mechanical seal - check face flatness
11. Reassemble in reverse order
12. Align pump and motor using laser alignment tool - max 0.05mm offset
13. Reconnect motor - verify rotation direction before starting
14. Start pump and monitor vibration for 30 minutes
15. Record final vibration readings in equipment history

ACCEPTANCE: Vibration < 2.5 mm/s RMS after running in
""",
        "metadata": {"category": "maintenance_procedure", "equipment": "pump", "location": "Plant B"},
    },
    {
        "id": "doc_pipeline_001",
        "content": """
EQUIPMENT: Process Pipeline P-104
MATERIAL: Carbon steel ASTM A106 Grade B
LOCATION: Plant A, Zone 1
SERVICE: High pressure steam condensate, 6 bar operating

SPECIFICATIONS:
- Nominal diameter: 4 inch (100mm)
- Wall thickness: 6mm (minimum allowable: 4.5mm)
- Design pressure: 10 bar
- Operating pressure: 4.0-6.5 bar (alarm: 7.0 bar, trip: 8.0 bar)
- Design temperature: 200°C
- Corrosion allowance: 1.5mm over 20-year design life
- Insulation: 50mm mineral wool, aluminium cladding

INSPECTION REQUIREMENTS:
- Visual inspection: Monthly
- Thickness measurement (UT): Every 6 months at designated inspection points
- Full radiographic inspection: Every 5 years
- Pressure test: After any modification or repair

MINIMUM WALL THICKNESS ALERT LEVELS:
- 5.5mm: Increased inspection frequency to monthly
- 5.0mm: Engineering assessment required
- 4.5mm: Immediate isolation and repair/replacement

PRESSURE RELIEF: PRV-301 set at 8.5 bar, tested annually
ISOLATION: Manual gate valve GV-101 (inlet), GV-102 (outlet)
""",
        "metadata": {"category": "equipment_spec", "equipment": "pipeline", "location": "Plant A"},
    },
    {
        "id": "doc_safety_001",
        "content": """
SAFETY PROCEDURE: Lockout/Tagout (LOTO) - Electrical Equipment
DOCUMENT: SP-SAFETY-001
REVISION: 6
MANDATORY for all maintenance activities on electrical equipment

LOTO PROCEDURE - 6 STEPS:
1. NOTIFY: Inform all affected personnel of LOTO activity and expected duration
2. IDENTIFY: Locate all energy sources (electrical, pneumatic, hydraulic, gravity)
3. ISOLATE: Open all disconnect switches, close all isolation valves
4. LOCK: Apply personal lock to each isolation point - one lock per person working
5. TAG: Attach danger tag to each isolation point - DO NOT OPERATE
6. VERIFY: Test equipment to confirm it is de-energised before starting work

ENERGY SOURCES TO ISOLATE:
- Main electrical supply: MCC panel, circuit breaker OFF + locked
- Control power: Remove control fuses, label each fuse
- Pneumatic supply: Close air supply valve, bleed residual pressure
- Hydraulic supply: Close hydraulic isolation valve, release stored pressure

RETURNING EQUIPMENT TO SERVICE:
1. Verify all tools and personnel clear of equipment
2. Remove all personal locks - each person removes their own lock only
3. Remove tags and document in LOTO register
4. Close all access panels and guards
5. Notify control room before re-energising
6. Restore energy sources in reverse isolation order
7. Test equipment at no-load before returning to service

VIOLATIONS: LOTO violations are grounds for immediate dismissal.
""",
        "metadata": {"category": "safety_procedure", "equipment": "all", "location": "all"},
    },
    {
        "id": "doc_compliance_001",
        "content": """
COMPLIANCE DOCUMENT: Pressure Vessel Inspection Requirements
DOCUMENT: COMP-PV-001
STANDARD: IS 2825 (Indian Standard for Unfired Pressure Vessels)

STATUTORY REQUIREMENTS:
- All pressure vessels above 25 litres capacity require registration with Chief Inspector of Factories
- Annual inspection by a competent person certified by state government
- Certificate of Fitness must be displayed on or near the vessel
- Vessels must not operate beyond certified maximum allowable working pressure (MAWP)

INSPECTION CATEGORIES:
1. EXTERNAL INSPECTION (Annual):
   - Visual examination of shell, heads, nozzles, supports
   - Verify safety relief valves are certified and within test date
   - Inspect pressure gauges for calibration certificate

2. INTERNAL INSPECTION (Every 2 years):
   - Internal visual examination after cleaning
   - Ultrasonic thickness measurement at all inspection points
   - Magnetic particle testing at welds if corrosion found

3. HYDRAULIC PRESSURE TEST (Every 5 years or after repair):
   - Test pressure: 1.5 x MAWP
   - Hold period: 30 minutes minimum
   - No pressure drop permitted
   - Witnessed by competent person

NON-COMPLIANCE PENALTY: Operation of uncertified pressure vessel is a criminal offence under Factories Act 1948, Section 36.
""",
        "metadata": {"category": "compliance", "equipment": "pressure_vessel", "location": "all"},
    },
    {
        "id": "doc_troubleshoot_001",
        "content": """
TROUBLESHOOTING GUIDE: Low Coolant Flow Rate
DOCUMENT: TG-FLOW-001
APPLICABLE SENSOR: flow_001 (Coolant Flow Rate, Plant B Zone 3)
ALARM CONDITION: Flow < 35 L/min (normal: 45-75 L/min)

DIAGNOSTIC DECISION TREE:

STEP 1 - Check pump P-03 status
  If pump not running: Check MCC panel, restart pump P-03B (standby)
  If pump running at reduced speed: Check VFD-01 setpoint, increase if safe
  If pump running at full speed: Go to Step 2

STEP 2 - Check inlet conditions
  Measure suction pressure at PI-201 (should be > 2 bar)
  If suction pressure low: Check filter FT-201 for blockage
  If suction pressure normal: Go to Step 3

STEP 3 - Check filter FT-201
  Measure differential pressure across filter (normal < 0.3 bar)
  If dP > 0.5 bar: Filter blocked - switch to bypass, clean/replace filter element
  If dP normal: Go to Step 4

STEP 4 - Check for leaks
  Walk entire coolant circuit from pump P-03 to heat exchanger HX-01
  If leak found: Isolate section, repair, pressure test
  If no leak: Go to Step 5

STEP 5 - Check isolation valves
  Verify IV-301, IV-302, IV-303, IV-304 all fully open
  If any partially closed: Open fully and monitor flow recovery

STEP 6 - Mechanical inspection
  Suspect pump impeller wear or cavitation
  Arrange mechanical inspection of pump P-03

ESCALATION: If flow < 20 L/min and cannot be restored within 30 minutes,
initiate controlled shutdown of boiler B-01 to prevent thermal damage.
""",
        "metadata": {"category": "troubleshooting", "equipment": "flow_system", "location": "Plant B"},
    },
    {
        "id": "doc_troubleshoot_002",
        "content": """
TROUBLESHOOTING GUIDE: High Vibration on Pump P-03
DOCUMENT: TG-VIB-001
APPLICABLE SENSOR: vibration_001 (Pump Vibration, Plant B Zone 1)
ALARM CONDITION: Vibration > 6.0 mm/s RMS (normal: 0.5-4.0 mm/s)

IMMEDIATE ACTIONS (vibration 6-7.5 mm/s):
1. Reduce pump speed by 10% via VFD-01
2. Check if vibration reduces - if yes, cavitation or resonance suspected
3. Notify maintenance supervisor
4. Increase monitoring frequency to every 15 minutes

IMMEDIATE SHUTDOWN (vibration > 7.5 mm/s):
1. Start standby pump P-03B immediately
2. Reduce P-03 speed gradually to minimum
3. Stop P-03 - do not trip suddenly (water hammer risk)
4. Lock out P-03, tag for inspection
5. Notify maintenance engineer for bearing inspection

VIBRATION ANALYSIS:
- 1x RPM frequency dominant: Imbalance or misalignment
- 2x RPM frequency dominant: Misalignment (angular or parallel)
- High frequency > 1000 Hz: Bearing defect
- Broadband noise: Cavitation or flow turbulence

ROOT CAUSE ACTIONS:
- Imbalance: Balance impeller, check for buildup or erosion
- Misalignment: Laser alignment of pump-motor coupling
- Bearing defect: Replace bearings - see MP-PUMP-007
- Cavitation: Increase suction pressure, check NPSH, reduce flow rate

RETURN TO SERVICE: Vibration must be < 2.5 mm/s RMS after any bearing or alignment work.
""",
        "metadata": {"category": "troubleshooting", "equipment": "pump", "location": "Plant B"},
    },
]


def chunk_documents(documents: list, chunk_size: int = 800, chunk_overlap: int = 200):
    """Split documents into smaller chunks for better retrieval precision."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunked_ids = []
    chunked_texts = []
    chunked_metadatas = []

    for doc in documents:
        chunks = splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            chunked_ids.append(f"{doc['id']}_chunk_{i}")
            chunked_texts.append(chunk)
            chunked_metadatas.append({
                **doc["metadata"],
                "source_doc_id": doc["id"],
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

    return chunked_ids, chunked_texts, chunked_metadatas


def seed():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="industrial_knowledge_base",
        embedding_function=ef,
        metadata={"description": "Industrial IoT equipment manuals, maintenance procedures, safety protocols"},
    )

    # Clear existing
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Chunk and seed
    chunked_ids, chunked_texts, chunked_metadatas = chunk_documents(DOCUMENTS)

    collection.add(
        ids=chunked_ids,
        documents=chunked_texts,
        metadatas=chunked_metadatas,
    )

    print(f"Seeded {len(DOCUMENTS)} documents as {len(chunked_ids)} chunks into ChromaDB")
    for doc in DOCUMENTS:
        chunk_count = sum(1 for m in chunked_metadatas if m["source_doc_id"] == doc["id"])
        print(f"  {doc['id']} -> {chunk_count} chunks")


if __name__ == "__main__":
    seed()
