"""Domain A, Session 5: Format Conversion & Encoding.

Session structure (11 tasks):
  Seed 1-3:  Use provided tools (read_csv, write_json, compute_hash)
  Gap 1:     Base64 encode/decode with metadata (content type, original size)
  Gap 2:     XML to JSON conversion (attributes, nested elements, text content)
  Variant 1: Hex encoding — should REUSE gap_1's tool (different encoding param)
  Variant 2: Different XML structure — should REUSE gap_2's tool
  Compose 1: Convert XML to JSON then base64 encode the result
  Regress 1: Re-run base64 encoding — should still work
  Adversarial 1: Malformed XML with unclosed tags, CDATA, special entities
  Adversarial 2: Binary-like data in base64, padding edge cases
"""

from ...types import Task, TaskType, Session
from .seed_tools import SEED_TOOLS


# ── Test data ────────────────────────────────────────────────────────

SIMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<catalog>
  <book id="1">
    <title>Python Cookbook</title>
    <author>David Beazley</author>
    <price currency="USD">49.99</price>
  </book>
  <book id="2">
    <title>Fluent Python</title>
    <author>Luciano Ramalho</author>
    <price currency="USD">59.99</price>
  </book>
</catalog>"""

CONFIG_XML = """<?xml version="1.0"?>
<config>
  <database>
    <host>localhost</host>
    <port>5432</port>
    <name>myapp</name>
    <credentials>
      <username>admin</username>
      <password>secret123</password>
    </credentials>
  </database>
  <cache>
    <enabled>true</enabled>
    <ttl>3600</ttl>
  </cache>
</config>"""

MALFORMED_XML = """<?xml version="1.0"?>
<root>
  <item id="1">
    <name>Valid Item</name>
    <description><![CDATA[This contains <special> & "characters"]]></description>
  </item>
  <item id="2">
    <name>Unclosed Tag
    <value>42</value>
  </item>
  <item id="3">
    <name>Entity Test &amp; &lt;more&gt;</name>
    <data>Normal text</data>
  </item>
</root>"""

MIXED_CONTENT_XML = """<?xml version="1.0"?>
<messages>
  <message from="alice" to="bob" timestamp="2025-01-01T10:00:00">
    Hello Bob!
  </message>
  <message from="bob" to="alice" timestamp="2025-01-01T10:01:00">
    Hi Alice, how are you?
  </message>
</messages>"""


# ── Tasks ────────────────────────────────────────────────────────────

TASKS = [
    # Seed tasks — use provided tools
    Task(
        id="seed_1",
        description=(
            "Read this CSV and convert to JSON:\n"
            "format,size,encoding\nxml,1024,utf-8\njson,512,ascii"
        ),
        task_type=TaskType.SEED,
        expected=[
            {"format": "xml", "size": "1024", "encoding": "utf-8"},
            {"format": "json", "size": "512", "encoding": "ascii"},
        ],
    ),
    Task(
        id="seed_2",
        description='Compute the SHA-256 hash of the string "format_conversion".',
        task_type=TaskType.SEED,
    ),
    Task(
        id="seed_3",
        description=(
            'Convert to JSON:\n'
            '[{"input": "xml", "output": "json", "status": "ok"}]'
        ),
        task_type=TaskType.SEED,
    ),

    # Gap tasks — require new tools
    Task(
        id="gap_1",
        description=(
            "Base64-encode the given string and return a dict with:\n"
            "- 'encoded': the base64-encoded string\n"
            "- 'original_size': byte length of the original string\n"
            "- 'encoded_size': length of the encoded string\n"
            "- 'encoding': 'base64'\n\n"
            "Input string: 'Hello, World! This is a test of base64 encoding.'"
        ),
        task_type=TaskType.GAP,
        expected={
            "encoded": "SGVsbG8sIFdvcmxkISBUaGlzIGlzIGEgdGVzdCBvZiBiYXNlNjQgZW5jb2Rpbmcu",
            "original_size": 50,
            "encoded_size": 68,
            "encoding": "base64",
        },
        hidden_tests=[
            {
                "input": {"text": "", "encoding": "base64"},
                "expected": {"encoded": "", "original_size": 0, "encoded_size": 0, "encoding": "base64"},
            },
            {
                "input": {"text": "a", "encoding": "base64"},
                "expected": {"encoded": "YQ==", "original_size": 1, "encoded_size": 4, "encoding": "base64"},
            },
        ],
        adversarial_tests=[
            {"input": {"text": "\x00\x01\x02\xff", "encoding": "base64"}},
            {"input": {"text": "a" * 10000, "encoding": "base64"}},
        ],
    ),
    Task(
        id="gap_2",
        description=(
            "Convert this XML to a JSON-compatible dict. Rules:\n"
            "- Element attributes become keys prefixed with '@' (e.g., @id)\n"
            "- Text content uses the key '#text'\n"
            "- Repeated child elements become lists\n"
            "- Nested elements become nested dicts\n\n"
            f"{SIMPLE_XML}"
        ),
        task_type=TaskType.GAP,
        expected={
            "catalog": {
                "book": [
                    {
                        "@id": "1",
                        "title": "Python Cookbook",
                        "author": "David Beazley",
                        "price": {"@currency": "USD", "#text": "49.99"},
                    },
                    {
                        "@id": "2",
                        "title": "Fluent Python",
                        "author": "Luciano Ramalho",
                        "price": {"@currency": "USD", "#text": "59.99"},
                    },
                ],
            },
        },
        hidden_tests=[
            {
                "input": {"xml_string": "<root><item>text</item></root>"},
                "expected": {"root": {"item": "text"}},
            },
            {
                "input": {"xml_string": '<root attr="val"/>'},
                "expected": {"root": {"@attr": "val"}},
            },
        ],
        adversarial_tests=[
            {"input": {"xml_string": ""}},
            {"input": {"xml_string": "not xml at all"}},
            {"input": {"xml_string": "<root><a>1</a><a>2</a><a>3</a></root>"}},
        ],
    ),

    # Variant tasks — should REUSE tools from gap tasks
    Task(
        id="variant_1",
        description=(
            "Hex-encode the given string and return a dict with:\n"
            "- 'encoded': the hex-encoded string\n"
            "- 'original_size': byte length of the original\n"
            "- 'encoded_size': length of the hex string\n"
            "- 'encoding': 'hex'\n\n"
            "Input string: 'Hello'"
        ),
        task_type=TaskType.VARIANT,
        reuses_task="gap_1",
        expected={
            "encoded": "48656c6c6f",
            "original_size": 5,
            "encoded_size": 10,
            "encoding": "hex",
        },
    ),
    Task(
        id="variant_2",
        description=(
            "Convert this configuration XML to a JSON-compatible dict using the same rules "
            "(@ prefix for attributes, #text for text content, lists for repeated elements).\n\n"
            f"{CONFIG_XML}"
        ),
        task_type=TaskType.VARIANT,
        reuses_task="gap_2",
        expected={
            "config": {
                "database": {
                    "host": "localhost",
                    "port": "5432",
                    "name": "myapp",
                    "credentials": {
                        "username": "admin",
                        "password": "secret123",
                    },
                },
                "cache": {
                    "enabled": "true",
                    "ttl": "3600",
                },
            },
        },
    ),

    # Compose task — convert XML to JSON then base64 encode
    Task(
        id="compose_1",
        description=(
            "Convert this XML to JSON (as a string), then base64-encode the JSON string. "
            "Return the encoding metadata dict.\n\n"
            f"{MIXED_CONTENT_XML}"
        ),
        task_type=TaskType.COMPOSE,
        composes_tasks=["gap_2", "gap_1"],
    ),

    # Regress task — re-test encoding
    Task(
        id="regress_1",
        description=(
            "Base64-encode the string 'Regression test data 12345' and return the metadata dict."
        ),
        task_type=TaskType.REGRESS,
        reuses_task="gap_1",
        expected={
            "encoded": "UmVncmVzc2lvbiB0ZXN0IGRhdGEgMTIzNDU=",
            "original_size": 27,
            "encoded_size": 36,
            "encoding": "base64",
        },
    ),

    # Adversarial tasks — break naive implementations
    Task(
        id="adversarial_1",
        description=(
            "Convert this malformed XML to JSON. Handle CDATA sections, XML entities "
            "(&amp; &lt; &gt;), and recover gracefully from unclosed tags if possible.\n\n"
            f"{MALFORMED_XML}"
        ),
        task_type=TaskType.ADVERSARIAL,
        breaks_task="gap_2",
    ),
    Task(
        id="adversarial_2",
        description=(
            "Base64-encode these edge-case strings and return metadata for each:\n"
            "1. Empty string: ''\n"
            "2. Single byte that requires padding: 'A'\n"
            "3. Two bytes that require padding: 'AB'\n"
            "4. Exactly 3 bytes (no padding): 'ABC'\n"
            "5. String with null bytes: '\\x00\\x00\\x00'\n\n"
            "Return a list of metadata dicts."
        ),
        task_type=TaskType.ADVERSARIAL,
        breaks_task="gap_1",
        expected=[
            {"encoded": "", "original_size": 0, "encoded_size": 0, "encoding": "base64"},
            {"encoded": "QQ==", "original_size": 1, "encoded_size": 4, "encoding": "base64"},
            {"encoded": "QUI=", "original_size": 2, "encoded_size": 4, "encoding": "base64"},
            {"encoded": "QUJD", "original_size": 3, "encoded_size": 4, "encoding": "base64"},
            {"encoded": "AAAA", "original_size": 3, "encoded_size": 4, "encoding": "base64"},
        ],
    ),
]


def create_session() -> Session:
    return Session(
        id="data_transform_s5",
        name="Format Conversion & Encoding",
        domain="data_transform",
        tasks=TASKS,
        seed_tools=SEED_TOOLS,
        description="Tests tool creation for base64/hex encoding and XML-to-JSON conversion, "
                    "with reuse across encodings, composition, and adversarial edge cases.",
    )
