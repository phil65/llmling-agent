# Extracting Data with extract() and extract_multiple()

The extraction methods provide powerful ways to parse structured data from text,
constructing typed objects from natural language descriptions.
These methods can work with any Python class that has a proper constructor.

## Basic Usage

### Single Instance Extraction
```python
class Person:
    def __init__(self, name: str, age: int | None = None):
        self.name = name
        self.age = age

text = "John is a 30-year-old software developer."
person = await agent.talk.extract(text, as_type=Person)
print(f"Found: {person.name}, {person.age}")  # "John", 30
```

### Multiple Instances
```python
text = """
Team members:
- Alice (25) leads development
- Bob (32) handles testing
- Carol (28) manages deployment
"""
team = await agent.talk.extract_multiple(text, as_type=Person)
for member in team:
    print(f"Member: {member.name}, {member.age}")
```

## Extraction Modes

Both methods support two extraction modes:

### Structured Mode (Default)

```python
# More robust, single-round extraction
result = await agent.talk.extract(
    text,
    as_type=Person,
    mode="structured"  # default
)
```


## Working with Complex Types

### Nested Objects

```python
class Address:
    def __init__(self, street: str, city: str):
        self.street = street
        self.city = city

class Employee:
    def __init__(self, name: str, address: Address):
        self.name = name
        self.address = address

text = "Alice lives at 123 Main St in New York"
employee = await agent.talk.extract(text, as_type=Employee)
```


## Custom Prompts

You can provide custom prompts for more specific extraction:

```python
result = await agent.talk.extract(
    text,
    as_type=Person,
    prompt="""
    Extract person information focusing on:
    1. Full name (first and last)
    2. Exact age if mentioned
    3. Ignore titles or honorifics
    """
)
```

For multiple extraction:

```python
results = await agent.talk.extract_multiple(
    text,
    as_type=Person,
    prompt="""
    Extract all team members mentioned.
    Look for:
    - Names in any format
    - Ages if provided
    - Skip generic references
    """
)
```

## Technical Details: How It Works

### Structured Mode (Default)

The structured mode works by converting the target class's constructor into a Pydantic model and using it for validation:

1. **Constructor Analysis**
```python
# Your class
class Person:
    def __init__(self, name: str, age: int | None = None):
        self.name = name
        self.age = age

# Internally converted to Pydantic model
class PersonModel(BaseModel):
    name: str
    age: int | None = None
```

2. **Response Container**
```python
# For single extraction
class Extraction(BaseModel):
    instance: PersonModel

# For multiple extraction
class Extraction(BaseModel):
    instances: list[PersonModel]
```

3. **LLM Interaction**

- Single structured response from LLM
- LLM sees complete schema upfront
- Natural mapping of fields
- One-shot validation through Pydantic

4. **Instance Creation**

```python
# Convert validated model to target class
result = as_type(**model_instance.model_dump())
```

Benefits:
- Single round-trip to LLM
- Clear schema validation
- Better handling of complex types
- More robust extraction


2. **Constructor Tool**

```python
async def construct(**kwargs: Any) -> T:
    """Dynamic constructor tool."""
    return as_type(**kwargs)
```

3. **LLM Interaction**

- LLM makes function calls
- For multiple: Repeated calls to add instances
- Each call validated separately
- State maintained between calls

4. **Instance Creation**

- Through tool call execution
- One instance per call
- Results collected in list (for multiple)

Trade-offs:

- More flexible but less robust
- Multiple round-trips for multiple instances
- Harder to maintain complex state
- More points of potential failure


### When to Use Which

This is a tough question. The Tool call mechanism was implemented first by [MarvinAI](https://askmarvin.ai), but seems less robust
compared to our new StructuredResponse approach. It doesnt include a sequence of tool calls and will use the validation capabilites of Pydantic-AI
in case that is the provider chosen.
