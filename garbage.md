# Plugin API Reference

Complete API reference for LichtFeld Studio plugins.

---

## Registration

```python
import lichtfeld as lf

lf.register_class(cls)           # Register a Panel, Operator, or Menu class
lf.unregister_class(cls)         # Unregister a Panel, Operator, or Menu class
```

---

## Panel

```python
import lichtfeld as lf
# lf.ui.Panel is the base class for all panels
```

| Attribute | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | `module.qualname` | Unique panel identifier |
| `label` | `str` | `""` | Display name (`id` fallback when empty) |
| `space` | `lf.ui.PanelSpace` | `lf.ui.PanelSpace.MAIN_PANEL_TAB` | Panel space (see below) |
| `parent` | `str` | `""` | Parent panel id. Embeds as a collapsible section; embedded panels must not override `space` |
| `order` | `int` | `100` | Sort order (lower = higher) |
| `options` | `set[lf.ui.PanelOption]` | `set()` | `DEFAULT_CLOSED`, `HIDE_HEADER` |
| `poll_dependencies` | `set[lf.ui.PollDependency]` | `{SCENE, SELECTION, TRAINING}` | Which state changes trigger `poll()` |
| `size` | `tuple[float, float] \| None` | `None` | Initial width/height hint, mainly for floating panels |
| `template` | `str \| os.PathLike[str]` | `""` | Retained RML template. Use an absolute path for plugin-local files |
| `style` | `str` | `""` | Inline RCSS appended to the retained document |
| `height_mode` | `lf.ui.PanelHeightMode` | `lf.ui.PanelHeightMode.FILL` | `FILL` or `CONTENT` for retained panels |
| `update_interval_ms` | `int` | `100` | Cadence for retained/hybrid `on_update()` work |

| Method | Returns | Description |
|---|---|---|
| `poll(cls, context)` | `bool` | Classmethod. Show/hide condition |
| `draw(self, ui)` | `None` | Immediate-mode content |
| `on_bind_model(self, ctx)` | `None` | Bind retained data models before document load |
| `on_mount(self, doc)` | `None` | Called once after the retained document mounts |
| `on_unmount(self, doc)` | `None` | Called before the retained document is destroyed |
| `on_update(self, doc)` | `None \| bool` | Periodic retained update. Return `True` to mark content dirty |
| `on_scene_changed(self, doc)` | `None` | Called when the active scene generation changes |

Registering a panel with the same `id` as an existing panel replaces it (see [Panel replacement](getting-started.md#panel-replacement)).

`lf.ui.Panel` is unified: a panel can start as `draw(ui)` only and later add `template`, `style`, `height_mode`, or retained hooks without switching base classes or rewriting the panel body.

Panel definitions are validated during `lf.register_class()`. Invalid enum values, removed legacy field names, unsupported retained features on `VIEWPORT_OVERLAY`, or conflicting embedded-panel fields raise `ValueError`, `TypeError`, or `AttributeError`.

The panel API is strict in v1: use the enum values above, not string literals.

### Panel spaces

`MAIN_PANEL_TAB`, `SIDE_PANEL`, `VIEWPORT_OVERLAY`, `SCENE_HEADER`, `FLOATING`, `STATUS_BAR`

### Retained shell behavior

If a panel uses retained features and `template` is empty, LichtFeld selects a shell automatically:

- `FLOATING` -> `rmlui/floating_window.rml`
- `STATUS_BAR` -> `rmlui/status_bar_panel.rml`
- Other retained panel spaces -> `rmlui/docked_panel.rml`

Built-in template aliases:

- `builtin:docked-panel`
- `builtin:floating-window`
- `builtin:status-bar`

### Panel styling guide

| Goal | Use | Notes |
|---|---|---|
| Minimal panel | `draw(self, ui)` | No extra files needed |
| Light retained styling | `style` | Inline RCSS text, not a path |
| Full custom retained UI | `template` | Use an absolute path for plugin-local `.rml` |
| Hybrid panel | `template` plus `draw(ui)` | Render immediate content into `<div id="im-root"></div>` |

When a plugin-local template file such as `main_panel.rml` is present, LichtFeld automatically loads a sibling `main_panel.rcss` stylesheet if it exists.

---

## Operator

```python
from lfs_plugins.types import Operator, Event
```

Operator extends `PropertyGroup`, so it supports typed properties as class attributes.

| Attribute     | Type       | Description                              |
|---------------|------------|------------------------------------------|
| `label`       | `str`      | Display name                             |
| `description` | `str`      | Tooltip text                             |
| `options`     | `Set[str]` | `{'UNDO', 'BLOCKING'}`                   |

| Method                          | Returns | Description                       |
|---------------------------------|---------|-----------------------------------|
| `poll(cls, context)`            | `bool`  | Classmethod. Can the op run?      |
| `invoke(self, context, event)`  | `set`   | Called on trigger, can start modal|
| `execute(self, context)`        | `set`   | Synchronous execution             |
| `modal(self, context, event)`   | `set`   | Handle events in modal mode       |
| `cancel(self, context)`         | `None`  | Called on cancellation             |

### Return sets

`{"FINISHED"}`, `{"CANCELLED"}`, `{"RUNNING_MODAL"}`, `{"PASS_THROUGH"}`

Or dict form: `{"status": "FINISHED", "key": value, ...}`

---

## Event

```python
from lfs_plugins.types import Event
```

| Attribute        | Type    | Description                                    |
|------------------|---------|------------------------------------------------|
| `type`           | `str`   | `'MOUSEMOVE'`, `'LEFTMOUSE'`, `'RIGHTMOUSE'`, `'MIDDLEMOUSE'`, `'KEY_A'`-`'KEY_Z'`, `'WHEELUPMOUSE'`, `'WHEELDOWNMOUSE'`, `'ESC'`, `'RET'`, `'SPACE'` |
| `value`          | `str`   | `'PRESS'`, `'RELEASE'`, `'NOTHING'`            |
| `mouse_x`        | `float` | Mouse X (viewport coords)                     |
| `mouse_y`        | `float` | Mouse Y (viewport coords)                     |
| `mouse_region_x` | `float` | Mouse X relative to region                     |
| `mouse_region_y` | `float` | Mouse Y relative to region                     |
| `delta_x`        | `float` | Mouse delta X                                  |
| `delta_y`        | `float` | Mouse delta Y                                  |
| `scroll_x`       | `float` | Scroll X offset                                |
| `scroll_y`       | `float` | Scroll Y offset                                |
| `shift`          | `bool`  | Shift modifier                                 |
| `ctrl`           | `bool`  | Ctrl modifier                                  |
| `alt`            | `bool`  | Alt modifier                                   |
| `pressure`       | `float` | Tablet pressure (1.0 for mouse)                |
| `over_gui`       | `bool`  | Mouse is over GUI element                      |
| `key_code`       | `int`   | Key code (see `key_codes.hpp`)                 |

---

## Properties

```python
from lfs_plugins.props import (
    Property, FloatProperty, IntProperty, BoolProperty,
    StringProperty, EnumProperty, FloatVectorProperty,
    IntVectorProperty, TensorProperty, CollectionProperty,
    PointerProperty, PropertyGroup, PropSubtype,
)
```

### FloatProperty

```python
FloatProperty(
    default: float = 0.0,
    min: float = -inf,
    max: float = inf,
    step: float = 0.1,
    precision: int = 3,
    subtype: str = "",       # FACTOR, PERCENTAGE, ANGLE, TIME, DISTANCE, POWER
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### IntProperty

```python
IntProperty(
    default: int = 0,
    min: int = -2**31,
    max: int = 2**31 - 1,
    step: int = 1,
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### BoolProperty

```python
BoolProperty(
    default: bool = False,
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### StringProperty

```python
StringProperty(
    default: str = "",
    maxlen: int = 0,         # 0 = unlimited
    subtype: str = "",       # FILE_PATH, DIR_PATH, FILE_NAME
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### EnumProperty

```python
EnumProperty(
    items: list[tuple[str, str, str]] = [],  # (identifier, label, description)
    default: str = None,     # First item if None
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### FloatVectorProperty

```python
FloatVectorProperty(
    default: tuple = (0.0, 0.0, 0.0),
    size: int = 3,
    min: float = -inf,
    max: float = inf,
    subtype: str = "",       # COLOR, COLOR_GAMMA, TRANSLATION, DIRECTION,
                             # VELOCITY, ACCELERATION, XYZ, EULER, QUATERNION
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### IntVectorProperty

```python
IntVectorProperty(
    default: tuple = (0, 0, 0),
    size: int = 3,
    min: int = -2**31,
    max: int = 2**31 - 1,
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### TensorProperty

```python
TensorProperty(
    shape: tuple = (),       # Use -1 for variable dims, e.g. (-1, 3)
    dtype: str = "float32",
    device: str = "cuda",
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### CollectionProperty

```python
CollectionProperty(
    type: Type[PropertyGroup],  # Item type
    name: str = "",
    description: str = "",
)
```

| Method               | Returns           | Description              |
|----------------------|-------------------|--------------------------|
| `add()`              | `PropertyGroup`   | Add new item             |
| `remove(index)`      | `None`            | Remove by index          |
| `clear()`            | `None`            | Remove all items         |
| `move(from, to)`     | `None`            | Reorder items            |
| `__len__()`          | `int`             | Item count               |
| `__getitem__(index)` | `PropertyGroup`   | Access by index          |
| `__iter__()`         | `Iterator`        | Iterate items            |

### PointerProperty

```python
PointerProperty(
    type: Type[PropertyGroup],  # Referenced type
    name: str = "",
    description: str = "",
)
```

| Method           | Returns         | Description                    |
|------------------|-----------------|--------------------------------|
| `get_instance()` | `PropertyGroup` | Get or create referenced object|

### PropertyGroup

```python
from lfs_plugins.props import PropertyGroup
```

| Method                    | Returns                | Description                         |
|---------------------------|------------------------|-------------------------------------|
| `get_instance()`          | `cls`                  | Classmethod. Singleton access       |
| `add_property(name, prop)`| `None`                 | Add property at runtime             |
| `remove_property(name)`   | `None`                 | Remove runtime property             |
| `get_all_properties()`    | `dict[str, Property]`  | All properties (class + runtime)    |
| `get(prop_id)`            | `Any`                  | Get property value by name          |
| `set(prop_id, value)`     | `None`                 | Set property value by name          |

### PropSubtype constants

```python
from lfs_plugins.props import PropSubtype

PropSubtype.NONE             # ""
PropSubtype.FILE_PATH        # "FILE_PATH"
PropSubtype.DIR_PATH         # "DIR_PATH"
PropSubtype.FILE_NAME        # "FILE_NAME"
PropSubtype.COLOR            # "COLOR"
PropSubtype.COLOR_GAMMA      # "COLOR_GAMMA"
PropSubtype.TRANSLATION      # "TRANSLATION"
PropSubtype.DIRECTION        # "DIRECTION"
PropSubtype.VELOCITY         # "VELOCITY"
PropSubtype.ACCELERATION     # "ACCELERATION"
PropSubtype.XYZ              # "XYZ"
PropSubtype.EULER            # "EULER"
PropSubtype.QUATERNION       # "QUATERNION"
PropSubtype.AXISANGLE        # "AXISANGLE"
PropSubtype.ANGLE            # "ANGLE"
PropSubtype.FACTOR           # "FACTOR"
PropSubtype.PERCENTAGE       # "PERCENTAGE"
PropSubtype.TIME             # "TIME"
PropSubtype.DISTANCE         # "DISTANCE"
PropSubtype.POWER            # "POWER"
PropSubtype.TEMPERATURE      # "TEMPERATURE"
PropSubtype.PIXEL            # "PIXEL"
PropSubtype.UNSIGNED         # "UNSIGNED"
PropSubtype.LAYER            # "LAYER"
PropSubtype.LAYER_MEMBER     # "LAYER_MEMBER"
```

---

## ToolDef / ToolRegistry

### ToolDef

```python
from lfs_plugins.tool_defs.definition import ToolDef, SubmodeDef, PivotModeDef
```

```python
@dataclass(frozen=True)
class ToolDef:
    id: str                                      # Unique tool ID
    label: str                                   # Display label
    icon: str                                    # Icon name
    group: str = "default"                       # "select", "transform", "paint", "utility"
    order: int = 100                             # Sort order within group
    description: str = ""                        # Tooltip
    shortcut: str = ""                           # Keyboard shortcut
    gizmo: str = ""                              # "translate", "rotate", "scale", ""
    operator: str = ""                           # Operator to invoke on activation
    submodes: tuple[SubmodeDef, ...] = ()
    pivot_modes: tuple[PivotModeDef, ...] = ()
    poll: Callable[[Any], bool] | None = None    # Availability check
    plugin_name: str = ""                        # For custom icon loading
    plugin_path: str = ""                        # For custom icon loading
```

| Method                  | Returns | Description                          |
|-------------------------|---------|--------------------------------------|
| `can_activate(context)` | `bool`  | Check if tool can be activated       |
| `to_dict()`             | `dict`  | Convert to dict for C++ interop      |

### SubmodeDef

```python
@dataclass(frozen=True)
class SubmodeDef:
    id: str           # Unique submode ID
    label: str        # Display label
    icon: str         # Icon name
    shortcut: str = ""
```

### PivotModeDef

```python
@dataclass(frozen=True)
class PivotModeDef:
    id: str           # Unique pivot mode ID
    label: str        # Display label
    icon: str         # Icon name
```

### ToolRegistry

```python
from lfs_plugins.tools import ToolRegistry
```

| Method                     | Returns              | Description                              |
|----------------------------|----------------------|------------------------------------------|
| `register_tool(tool)`      | `None`               | Register a custom tool                   |
| `unregister_tool(tool_id)` | `None`               | Unregister by ID                         |
| `get(tool_id)`             | `Optional[ToolDef]`  | Get tool by ID (builtins first)          |
| `get_all()`                | `list[ToolDef]`      | All tools (builtins + custom, sorted)    |
| `set_active(tool_id)`      | `bool`               | Activate a tool                          |
| `get_active()`             | `Optional[ToolDef]`  | Get active tool                          |
| `get_active_id()`          | `str`                | Get active tool ID                       |

---

## Signals

```python
from lfs_plugins.ui.signals import Signal, ComputedSignal, ThrottledSignal, Batch, batch
```

### Signal[T]

```python
Signal(initial_value: T, name: str = "")
```

| Property/Method                          | Returns           | Description                      |
|------------------------------------------|-------------------|----------------------------------|
| `.value`                                 | `T`               | Get/set current value            |
| `.peek()`                                | `T`               | Get without tracking             |
| `.subscribe(callback)`                   | `() -> None`      | Subscribe; returns unsubscribe fn|
| `.subscribe_as(owner, callback)`         | `() -> None`      | Owner-tracked subscription       |

### ComputedSignal[T]

```python
ComputedSignal(compute: Callable[[], T], dependencies: list[Signal])
```

| Property/Method                          | Returns           | Description                      |
|------------------------------------------|-------------------|----------------------------------|
| `.value`                                 | `T`               | Get computed value (lazy)        |
| `.subscribe(callback)`                   | `() -> None`      | Subscribe to changes             |
| `.subscribe_as(owner, callback)`         | `() -> None`      | Owner-tracked subscription       |

### ThrottledSignal[T]

```python
ThrottledSignal(initial_value: T, max_rate_hz: float = 60.0, name: str = "")
```

| Property/Method                          | Returns           | Description                      |
|------------------------------------------|-------------------|----------------------------------|
| `.value`                                 | `T`               | Get/set current value            |
| `.flush()`                               | `None`            | Force pending notification       |
| `.subscribe(callback)`                   | `() -> None`      | Subscribe to changes             |
| `.subscribe_as(owner, callback)`         | `() -> None`      | Owner-tracked subscription       |

### Batch / batch()

```python
with Batch():       # Class form
    ...

with batch():       # Function form
    ...
```

Defers all signal notifications until the block exits.

### SubscriptionRegistry

```python
from lfs_plugins.ui.subscription_registry import SubscriptionRegistry

registry = SubscriptionRegistry.instance()
unsub = registry.register(owner="my_plugin", unsubscribe_fn=fn)
registry.unregister_all("my_plugin")   # Cleanup on unload
```

---

## Capabilities

```python
from lfs_plugins.capabilities import CapabilityRegistry, CapabilitySchema, Capability
from lfs_plugins.context import PluginContext, SceneContext, ViewContext, CapabilityBroker
```

### CapabilityRegistry

```python
registry = CapabilityRegistry.instance()
```

| Method                                    | Returns              | Description                          |
|-------------------------------------------|----------------------|--------------------------------------|
| `register(name, handler, ...)`            | `None`               | Register a capability                |
| `unregister(name)`                        | `bool`               | Unregister by name                   |
| `unregister_all_for_plugin(plugin_name)`  | `int`                | Unregister all for plugin            |
| `invoke(name, args)`                      | `dict`               | Invoke capability                    |
| `get(name)`                               | `Optional[Capability]`| Get by name                         |
| `list_all()`                              | `list[Capability]`   | List all capabilities                |
| `has(name)`                               | `bool`               | Check existence                      |

#### register() parameters

```python
registry.register(
    name: str,                    # Unique name, e.g. "my_plugin.feature"
    handler: Callable,            # fn(args: dict, ctx: PluginContext) -> dict
    description: str = "",
    schema: CapabilitySchema = None,
    plugin_name: str = None,
    requires_gui: bool = True,
)
```

### CapabilitySchema

```python
@dataclass
class CapabilitySchema:
    properties: dict[str, dict[str, Any]]   # JSON Schema-like property defs
    required: list[str]                      # Required property names
```

### Capability

```python
@dataclass
class Capability:
    name: str
    description: str
    handler: Callable
    schema: CapabilitySchema
    plugin_name: Optional[str]
    requires_gui: bool
```

### PluginContext

```python
@dataclass
class PluginContext:
    scene: Optional[SceneContext]
    view: Optional[ViewContext]
    capabilities: CapabilityBroker
```

| Method                               | Returns         | Description                    |
|--------------------------------------|-----------------|--------------------------------|
| `build(registry, include_view=True)` | `PluginContext`  | Classmethod. Build from state  |

### SceneContext

```python
@dataclass
class SceneContext:
    scene: Any                          # PyScene object
```

| Method                       | Returns | Description                  |
|------------------------------|---------|------------------------------|
| `set_selection_mask(mask)`   | `None`  | Apply selection mask         |

### ViewContext

```python
@dataclass
class ViewContext:
    image: Any                          # [H, W, 3] tensor
    screen_positions: Optional[Any]     # [N, 2] tensor or None
    width: int
    height: int
    fov: float
    rotation: Any                       # [3, 3] tensor
    translation: Any                    # [3] tensor
```

### CapabilityBroker

```python
class CapabilityBroker:
    def invoke(self, name: str, args: dict = None) -> dict
    def has(self, name: str) -> bool
    def list_all(self) -> list[str]
```

---

## PluginManager

```python
from lfs_plugins.manager import PluginManager
```

```python
mgr = PluginManager.instance()
```

| Method                                | Returns                    | Description                       |
|---------------------------------------|----------------------------|-----------------------------------|
| `plugins_dir`                         | `Path`                     | Property. `~/.lichtfeld/plugins/` |
| `discover()`                          | `list[PluginInfo]`         | Scan for plugins                  |
| `load(name, on_progress=None)`        | `bool`                     | Load a plugin                     |
| `unload(name)`                        | `bool`                     | Unload a plugin                   |
| `reload(name)`                        | `bool`                     | Hot-reload a plugin               |
| `load_all()`                          | `dict[str, bool]`          | Load all user-enabled plugins     |
| `install(url, on_progress=None, auto_load=True)` | `str`          | Install from Git URL              |
| `uninstall(name)`                     | `bool`                     | Remove a plugin                   |
| `update(name, on_progress=None)`      | `bool`                     | Update a plugin                   |
| `search(query, compatible_only=True)` | `list[RegistryPluginInfo]` | Search registry                   |
| `check_updates()`                     | `dict[str, tuple]`         | Check installed plugin updates    |
| `get_state(name)`                     | `Optional[PluginState]`    | Get plugin state                  |
| `get_error(name)`                     | `Optional[str]`            | Get error message                 |
| `get_traceback(name)`                 | `Optional[str]`            | Get error traceback               |

### PluginInfo

```python
@dataclass
class PluginInfo:
    name: str
    version: str
    path: Path
    description: str = ""
    author: str = ""
    entry_point: str = "__init__"
    dependencies: list[str] = []
    auto_start: bool = False
    hot_reload: bool = True
    plugin_api: str = ""
    lichtfeld_version: str = ""
    required_features: list[str] = []
```

### PluginState

```python
class PluginState(Enum):
    UNLOADED = "unloaded"
    INSTALLING = "installing"
    LOADING = "loading"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"
```

### `lichtfeld.plugins` convenience API

```python
import lichtfeld as lf
```

| Function | Returns | Description |
|---|---|---|
| `lf.plugins.discover()` | `list[PluginInfo]` | Discover plugins in `~/.lichtfeld/plugins/` |
| `lf.plugins.load(name)` | `bool` | Load a plugin |
| `lf.plugins.unload(name)` | `bool` | Unload a plugin |
| `lf.plugins.reload(name)` | `bool` | Reload a plugin |
| `lf.plugins.load_all()` | `dict[str, bool]` | Load all user-enabled plugins |
| `lf.plugins.start_watcher()` | `None` | Start the hot-reload watcher |
| `lf.plugins.stop_watcher()` | `None` | Stop the hot-reload watcher |
| `lf.plugins.get_state(name)` | `PluginState \| None` | Read plugin state |
| `lf.plugins.get_error(name)` | `str \| None` | Read the last plugin error |
| `lf.plugins.get_traceback(name)` | `str \| None` | Read the full traceback |
| `lf.plugins.create(name)` | `str` | Create the v1 source scaffold in `~/.lichtfeld/plugins/<name>` |

`lf.plugins.create()` writes the source package, including `panels/main_panel.py`, `panels/main_panel.rml`, and `panels/main_panel.rcss`. If you want a scaffold that also adds `.venv`, `.vscode`, and `pyrightconfig.json`, use the CLI command `LichtFeld-Studio plugin create <name>`.

Runtime compatibility constants:

| Constant | Type | Description |
|---|---|---|
| `lf.PLUGIN_API_VERSION` | `str` | Host plugin API version |
| `lf.plugins.API_VERSION` | `str` | Same plugin API version through the plugin namespace |
| `lf.plugins.FEATURES` | `list[str]` | Supported optional plugin features on this host |

---

## Layout API

The `ui` object passed to `Panel.draw()` provides the immediate widget API used by both simple and hybrid panels. Depending on the panel space and shell, it may be rendered through the direct viewport path or the immediate-mode RML bridge, but the Python widget surface stays the same.

### Text

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `label(text)`                                       | `None`  | Plain text               |
| `label_centered(text)`                              | `None`  | Centered text            |
| `heading(text)`                                     | `None`  | Large heading            |
| `text_colored(text, color)`                         | `None`  | Colored text (RGBA tuple)|
| `text_colored_centered(text, color)`                | `None`  | Centered colored text    |
| `text_selectable(text, height=0)`                   | `None`  | Selectable text          |
| `text_wrapped(text)`                                | `None`  | Word-wrapped text        |
| `text_disabled(text)`                               | `None`  | Grayed-out text          |
| `bullet_text(text)`                                 | `None`  | Bulleted text            |

### Buttons

| Method                                              | Returns | Description                    |
|-----------------------------------------------------|---------|--------------------------------|
| `button(label, size=(0,0))`                         | `bool`  | Standard button                |
| `button_styled(label, style, size=(0,0))`           | `bool`  | Styled: "success", "error", "warning", "primary", "secondary" |
| `button_callback(label, callback=None, size=(0,0))` | `bool`  | Button with callback           |
| `small_button(label)`                               | `bool`  | Compact button                 |
| `invisible_button(id, size)`                        | `bool`  | Invisible clickable area       |

### Input

| Method                                                              | Returns             | Description              |
|---------------------------------------------------------------------|---------------------|--------------------------|
| `checkbox(label, value)`                                            | `(bool, bool)`      | (changed, new_value)     |
| `radio_button(label, current, value)`                               | `(bool, int)`       | Radio button             |
| `input_text(label, value)`                                          | `(bool, str)`       | Text input               |
| `input_text_with_hint(label, hint, value)`                          | `(bool, str)`       | Text with placeholder    |
| `input_text_enter(label, value)`                                    | `(bool, str)`       | Confirm on Enter         |
| `input_float(label, value, step=0, step_fast=0, format='%.3f')`    | `(bool, float)`     | Float input              |
| `input_int(label, value, step=1, step_fast=100)`                    | `(bool, int)`       | Integer input            |
| `input_int_formatted(label, value, step=0, step_fast=0)`           | `(bool, int)`       | Formatted int input      |

### Sliders & Drags

| Method                                                              | Returns              | Description           |
|---------------------------------------------------------------------|----------------------|-----------------------|
| `slider_float(label, value, min, max)`                              | `(bool, float)`      | Float slider          |
| `slider_int(label, value, min, max)`                                | `(bool, int)`        | Integer slider        |
| `slider_float2(label, value, min, max)`                             | `(bool, tuple)`      | 2-component slider    |
| `slider_float3(label, value, min, max)`                             | `(bool, tuple)`      | 3-component slider    |
| `drag_float(label, value, speed=1, min=0, max=0)`                  | `(bool, float)`      | Float drag            |
| `drag_int(label, value, speed=1, min=0, max=0)`                    | `(bool, int)`        | Integer drag          |

### Selection

| Method                                                              | Returns              | Description              |
|---------------------------------------------------------------------|----------------------|--------------------------|
| `combo(label, current_idx, items)`                                  | `(bool, int)`        | Dropdown selector        |
| `listbox(label, current_idx, items, height_items=-1)`               | `(bool, int)`        | List selector            |
| `selectable(label, selected=False, height=0)`                       | `bool`               | Selectable item          |
| `prop_search(data, prop_id, search_data, search_prop, text='')`     | `(bool, int)`        | Searchable dropdown      |

### Color

| Method                                              | Returns              | Description              |
|-----------------------------------------------------|----------------------|--------------------------|
| `color_edit3(label, color)`                         | `(bool, tuple)`      | RGB color picker         |
| `color_edit4(label, color)`                         | `(bool, tuple)`      | RGBA color picker        |
| `color_button(label, color, size=(0,0))`            | `bool`               | Color swatch button      |

### File/Path

| Method                                              | Returns              | Description              |
|-----------------------------------------------------|----------------------|--------------------------|
| `path_input(label, value, folder_mode=True, dialog_title='')` | `(bool, str)` | File/folder picker  |

### Property Binding

| Method                                              | Returns              | Description                              |
|-----------------------------------------------------|----------------------|------------------------------------------|
| `prop(data, prop_id, text=None)`                    | `(bool, Any)`        | Auto-widget based on property type       |

### Layout Structure

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `separator()`                                       | `None`  | Horizontal line          |
| `spacing()`                                         | `None`  | Vertical space           |
| `same_line(offset=0, spacing=-1)`                   | `None`  | Next widget on same line |
| `new_line()`                                        | `None`  | Force new line           |
| `indent(width=0)`                                   | `None`  | Increase indent          |
| `unindent(width=0)`                                 | `None`  | Decrease indent          |
| `begin_group()` / `end_group()`                     | `None`  | Logical widget group     |
| `set_next_item_width(width)`                        | `None`  | Width for next widget    |
| `dummy(size)`                                       | `None`  | Empty space placeholder  |

### Collapsible / Tree

| Method                                              | Returns | Description                    |
|-----------------------------------------------------|---------|--------------------------------|
| `collapsing_header(label, default_open=False)`      | `bool`  | Collapsible section            |
| `tree_node(label)`                                  | `bool`  | Tree node (call `tree_pop()`)  |
| `tree_node_ex(label, flags='')`                     | `bool`  | Extended tree node             |
| `tree_pop()`                                        | `None`  | Close tree node                |

### Tables

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `begin_table(id, columns)`                          | `bool`  | Start table              |
| `table_setup_column(label, width=0)`                | `None`  | Define column            |
| `table_headers_row()`                               | `None`  | Draw header row          |
| `table_next_row()`                                  | `None`  | Next row                 |
| `table_next_column()`                               | `None`  | Next column              |
| `table_set_column_index(column)`                    | `bool`  | Jump to column           |
| `table_set_bg_color(target, color)`                 | `None`  | Set row/cell background  |
| `end_table()`                                       | `None`  | End table                |

### Images

| Method                                                       | Returns | Description            |
|--------------------------------------------------------------|---------|------------------------|
| `image(texture_id, size, tint=(1,1,1,1))`                   | `None`  | Display image          |
| `image_uv(texture_id, size, uv0, uv1, tint=(1,1,1,1))`     | `None`  | Image with UV coords   |
| `image_button(id, texture_id, size, tint=(1,1,1,1))`        | `bool`  | Clickable image        |
| `toolbar_button(id, tex, size, selected=F, disabled=F, tooltip='')` | `bool` | Toolbar icon button |
| `image_tensor(label, tensor, size, tint=None)`               | `None`  | Display a tensor as an image (cached by label) |
| `image_texture(texture, size, tint=None)`                    | `None`  | Display a `DynamicTexture` |

`image_tensor` is the simplest way to display a GPU tensor — it internally manages a `DynamicTexture` cached by `label`. The tensor must be `[H, W, 3]` or `[H, W, 4]` (RGB/RGBA). CPU tensors and non-float32 dtypes are converted automatically.

```python
ui.image_tensor("preview", my_tensor, (256, 256))
```

For full control (e.g. reusing one texture across multiple draw calls), use `DynamicTexture` directly:

```python
tex = lf.ui.DynamicTexture(tensor)   # or DynamicTexture() + tex.update(tensor)
ui.image_texture(tex, (256, 256))
```

---

### DynamicTexture

GPU tensor to OpenGL texture bridge via CUDA-GL interop.

```python
tex = lf.ui.DynamicTexture()          # Empty
tex = lf.ui.DynamicTexture(tensor)    # From tensor
```

| Method / Property  | Returns              | Description                                    |
|--------------------|----------------------|------------------------------------------------|
| `update(tensor)`   | `None`               | Upload `[H, W, 3\|4]` tensor (auto-converts CPU→CUDA, uint8→float32) |
| `destroy()`        | `None`               | Release GL resources                           |
| `id`               | `int`                | OpenGL texture ID                              |
| `width`            | `int`                | Current width in pixels                        |
| `height`           | `int`                | Current height in pixels                       |
| `valid`            | `bool`               | `True` if texture is initialized               |
| `uv1`             | `tuple[float, float]` | UV scale factors for power-of-2 padding        |

Calling `update()` with a different resolution automatically recreates the GL texture. Textures are freed on plugin unload via `lf.ui.free_plugin_textures(name)`.

### Drag & Drop

| Method                                              | Returns       | Description              |
|-----------------------------------------------------|---------------|--------------------------|
| `begin_drag_drop_source()`                          | `bool`        | Start drag source        |
| `set_drag_drop_payload(type, data)`                 | `None`        | Set drag payload         |
| `end_drag_drop_source()`                            | `None`        | End drag source          |
| `begin_drag_drop_target()`                          | `bool`        | Start drag target        |
| `accept_drag_drop_payload(type)`                    | `str or None` | Accept payload           |
| `end_drag_drop_target()`                            | `None`        | End drag target          |

### Popups & Menus

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `begin_popup(id)`                                   | `bool`  | Start popup              |
| `begin_context_menu(id='')`                         | `bool`  | Styled context menu      |
| `begin_popup_modal(title)`                          | `bool`  | Modal popup              |
| `open_popup(id)`                                    | `None`  | Trigger popup open       |
| `end_popup()` / `end_popup_modal()`                 | `None`  | End popup/modal          |
| `end_context_menu()`                                | `None`  | End context menu         |
| `close_current_popup()`                             | `None`  | Close current popup      |
| `begin_menu(label)`                                 | `bool`  | Start menu               |
| `end_menu()`                                        | `None`  | End menu                 |
| `begin_menu_bar()` / `end_menu_bar()`               | `bool`  | Menu bar                 |
| `menu_item(label, enabled=True)`                    | `bool`  | Menu item                |
| `menu_item_toggle(label, shortcut, selected)`       | `bool`  | Toggle menu item         |
| `menu_item_shortcut(label, shortcut, enabled=True)` | `bool`  | Menu item with shortcut  |
| `menu(menu_id, text='', icon='')`                   | `None`  | Inline menu reference    |
| `popover(panel_id, text='', icon='')`               | `None`  | Panel popover            |

### Windows & Children

| Method                                                        | Returns        | Description           |
|---------------------------------------------------------------|----------------|-----------------------|
| `begin_window(title, flags=0)`                                | `bool`         | Start window          |
| `begin_window_closable(title, flags=0)`                       | `(bool, bool)` | Closable window       |
| `end_window()`                                                | `None`         | End window            |
| `begin_child(id, size=(0,0), border=False)`                   | `bool`         | Start child region    |
| `end_child()`                                                 | `None`         | End child region      |
| `set_next_window_pos(pos, first_use=False)`                   | `None`         | Set window position   |
| `set_next_window_size(size, first_use=False)`                 | `None`         | Set window size       |
| `set_next_window_pos_center()`                                | `None`         | Center window         |
| `set_next_window_pos_centered(first_use=False)`               | `None`         | Center next window (main viewport) |
| `set_next_window_pos_viewport_center()`                       | `None`         | Viewport center       |
| `set_next_window_focus()`                                     | `None`         | Focus next window     |
| `set_next_window_bg_alpha(alpha)`                             | `None`         | Set next window BG alpha |
| `push_window_style()` / `pop_window_style()`                  | `None`         | Window style stack    |
| `push_modal_style()` / `pop_modal_style()`                    | `None`         | Modal style stack     |

### Drawing (Viewport)

| Method                                                         | Returns | Description            |
|----------------------------------------------------------------|---------|------------------------|
| `draw_line(x0, y0, x1, y1, color, thickness=1)`               | `None`  | Line                   |
| `draw_rect(x0, y0, x1, y1, color, thickness=1)`               | `None`  | Rectangle outline      |
| `draw_rect_filled(x0, y0, x1, y1, color, bg=False)`           | `None`  | Filled rectangle       |
| `draw_rect_rounded(x0, y0, x1, y1, color, r, thick=1, bg=F)`  | `None`  | Rounded rect outline   |
| `draw_rect_rounded_filled(x0, y0, x1, y1, color, r, bg=F)`    | `None`  | Filled rounded rect    |
| `draw_circle(x, y, radius, color, segments=32, thickness=1)`  | `None`  | Circle outline         |
| `draw_circle_filled(x, y, radius, color, segments=32)`        | `None`  | Filled circle          |
| `draw_triangle_filled(x0, y0, x1, y1, x2, y2, color, bg=F)`  | `None`  | Filled triangle        |
| `draw_text(x, y, text, color, bg=False)`                      | `None`  | Text at position       |
| `draw_polyline(points, color, closed=False, thickness=1)`      | `None`  | Polyline               |
| `draw_poly_filled(points, color)`                              | `None`  | Filled polygon         |
| `plot_lines(label, values, scale_min, scale_max, size)`        | `None`  | Line plot              |

### Drawing (Window)

| Method                                                         | Returns | Description            |
|----------------------------------------------------------------|---------|------------------------|
| `draw_window_rect_filled(x0, y0, x1, y1, color)`              | `None`  | Filled rect to window  |
| `draw_window_rect(x0, y0, x1, y1, color, thickness=1)`        | `None`  | Rect outline to window |
| `draw_window_line(x0, y0, x1, y1, color, thickness=1)`        | `None`  | Line to window         |
| `draw_window_text(x, y, text, color)`                          | `None`  | Text to window         |
| `draw_window_triangle_filled(x0, y0, x1, y1, x2, y2, color)` | `None`  | Triangle to window     |

### Progress & Status

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `progress_bar(fraction, overlay='', width=0)`       | `None`  | Progress bar             |
| `set_tooltip(text)`                                 | `None`  | Tooltip for last item    |

### State Queries

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `is_item_hovered()`                                 | `bool`  | Last item hovered        |
| `is_item_clicked(button=0)`                         | `bool`  | Last item clicked        |
| `is_item_active()`                                  | `bool`  | Last item active         |
| `is_window_focused()`                               | `bool`  | Window has focus         |
| `is_window_hovered()`                               | `bool`  | Window is hovered        |
| `is_mouse_double_clicked(button=0)`                 | `bool`  | Double click detected    |
| `is_mouse_dragging(button=0)`                       | `bool`  | Mouse dragging           |
| `get_mouse_wheel()`                                 | `float` | Scroll wheel delta       |
| `get_mouse_delta()`                                 | `tuple` | Mouse delta (dx, dy)     |

### Position / Size

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `get_cursor_pos()`                                  | `tuple` | Cursor position          |
| `get_cursor_screen_pos()`                           | `tuple` | Cursor screen position   |
| `get_window_pos()`                                  | `tuple` | Window position          |
| `get_window_width()`                                | `float` | Window width             |
| `get_text_line_height()`                            | `float` | Text line height         |
| `get_content_region_avail()`                        | `tuple` | Available content area   |
| `get_viewport_pos()`                                | `tuple` | Viewport position        |
| `get_viewport_size()`                               | `tuple` | Viewport size            |
| `get_dpi_scale()`                                   | `float` | DPI scale factor         |
| `calc_text_size(text)`                              | `tuple` | Text dimensions          |

### Styling

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `push_style_var(var, value)`                        | `None`  | Push float style var     |
| `push_style_var_vec2(var, value)`                   | `None`  | Push vec2 style var      |
| `pop_style_var(count=1)`                            | `None`  | Pop style vars           |
| `push_style_color(col, color)`                      | `None`  | Push color override      |
| `pop_style_color(count=1)`                          | `None`  | Pop color overrides      |
| `push_item_width(width)` / `pop_item_width()`      | `None`  | Item width stack         |
| `begin_disabled(disabled=True)` / `end_disabled()`  | `None`  | Disable widget region. For composable disabled regions, prefer `SubLayout.enabled` (see Layout Composition below). |

### Keyboard / Mouse Capture

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `set_keyboard_focus_here()`                         | `None`  | Focus next widget        |
| `capture_keyboard_from_app(capture=True)`           | `None`  | Capture keyboard input   |
| `capture_mouse_from_app(capture=True)`              | `None`  | Capture mouse input      |
| `set_mouse_cursor_hand()`                           | `None`  | Set hand cursor          |

### Cursor Control

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `set_cursor_pos(pos)`                               | `None`  | Set cursor position      |
| `set_cursor_pos_x(x)`                               | `None`  | Set cursor X             |
| `set_scroll_here_y(ratio=0.5)`                      | `None`  | Scroll to current Y      |

### Specialized Widgets

| Method                                                                            | Returns        | Description           |
|-----------------------------------------------------------------------------------|----------------|-----------------------|
| `crf_curve_preview(label, gamma, toe, shoulder, gamma_r=0, gamma_g=0, gamma_b=0)`| `None`         | Tone curve preview    |
| `chromaticity_diagram(label, rx, ry, gx, gy, bx, by, nx, ny, range=0.5)`         | `(bool, list)` | Chromaticity diagram  |
| `template_list(list_type_id, list_id, data, prop_id, active_data, active_prop, rows=5)` | `(int, int)` | Custom list template |

### Layout Composition

Create composable sub-layouts with automatic widget positioning and state cascading.

| Method                                             | Returns     | Description                     |
|----------------------------------------------------|-------------|---------------------------------|
| `row()`                                            | `SubLayout` | Horizontal layout               |
| `column()`                                         | `SubLayout` | Vertical layout                 |
| `split(factor=0.5)`                                | `SubLayout` | Two-column split                |
| `box()`                                            | `SubLayout` | Bordered container              |
| `grid_flow(columns=0, even_columns=True, even_rows=True)` | `SubLayout` | Responsive grid         |
| `prop_enum(data, prop_id, value, text='')`          | `bool`      | Enum toggle button              |

`SubLayout` is a context manager. Use `with ui.row() as row:` to enter the layout, then call widget methods on `row` instead of `ui`. Sub-layouts nest arbitrarily.

#### SubLayout state properties

| Property  | Type    | Description                                |
|-----------|---------|--------------------------------------------|
| `enabled` | `bool`  | Disabled state (cascades to children)      |
| `active`  | `bool`  | Active state (cascades to children)        |
| `alert`   | `bool`  | One-shot alert styling (red text/bg)       |

#### Example

```python
def draw(self, ui):
    with ui.row() as row:
        row.prop_enum(self, "mode", "fast", "Fast")
        row.prop_enum(self, "mode", "quality", "Quality")

    with ui.box() as box:
        box.heading("Settings")
        box.prop(self, "opacity")

    with ui.column() as col:
        col.enabled = self.is_active
        col.prop(self, "value")
        with col.row() as row:
            row.button("Apply")
            row.button("Cancel")

    with ui.grid_flow(columns=3) as grid:
        for item in items:
            with grid.box() as cell:
                cell.label(item.name)
                cell.button("Select")
```

---

## Scene API (lf module)

```python
import lichtfeld as lf
```

### Scene Management

| Function                  | Returns          | Description                       |
|---------------------------|------------------|-----------------------------------|
| `get_scene()`             | `Scene or None`  | Get scene object                  |
| `get_render_scene()`      | `Scene or None`  | Get render scene (PyScene)        |
| `has_scene()`             | `bool`           | Whether scene is loaded           |
| `clear_scene()`           | `None`           | Clear all scene content           |
| `load_file(path, is_dataset=False)` | `None` | Load PLY or dataset               |
| `load_config_file(path)`  | `None`           | Load JSON config                  |
| `get_scene_generation()`  | `int`            | Scene generation counter          |
| `list_scene()`            | `None`           | Print scene tree                  |

### Node Operations (on Scene object)

| Method                                            | Returns          | Description                       |
|---------------------------------------------------|------------------|-----------------------------------|
| `add_group(name, parent=-1)`                      | `int`            | Add group node                    |
| `add_splat(name, means, sh0, shN, scaling, rotation, opacity, ...)` | `int` | Add splat node       |
| `add_point_cloud(name, points, colors, parent=-1)`| `int`            | Add point cloud                   |
| `add_camera(name, parent, R, T, fx, fy, w, h, ...)` | `int`         | Add camera node                   |
| `remove_node(name, keep_children=False)`          | `None`           | Remove node                       |
| `rename_node(old, new)`                           | `bool`           | Rename node                       |
| `reparent(node_id, new_parent_id)`                | `None`           | Change parent                     |
| `duplicate_node(name)`                            | `str`            | Duplicate, returns new name       |
| `merge_group(group_name)`                         | `str`            | Merge group children              |
| `get_node(name)`                                  | `SceneNode`      | Get node by name                  |
| `get_node_by_id(id)`                              | `SceneNode`      | Get node by ID                    |
| `get_nodes()`                                     | `list[SceneNode]`| All nodes                         |
| `get_visible_nodes()`                             | `list[SceneNode]`| Visible nodes only                |
| `root_nodes()`                                    | `list[int]`      | Root node IDs                     |
| `is_node_effectively_visible(id)`                 | `bool`           | Considers parent visibility       |
| `total_gaussian_count`                            | `int`            | Property. Total gaussians         |
| `invalidate_cache()`                              | `None`           | Clear internal cache (no redraw)  |
| `notify_changed()`                                | `None`           | Invalidate cache + trigger viewport redraw |

### SceneNode Properties

| Property/Method     | Returns              | Description                       |
|---------------------|----------------------|-----------------------------------|
| `id`                | `int`                | Node ID                           |
| `name`              | `str`                | Node name                         |
| `type`              | `NodeType`           | Node type enum (SPLAT, POINTCLOUD, GROUP, etc.) |
| `parent_id`         | `int`                | Parent node ID (-1 for root)      |
| `children`          | `list[int]`          | Child node IDs                    |
| `visible`           | `bool`               | Visibility flag                   |
| `locked`            | `bool`               | Lock flag                         |
| `gaussian_count`    | `int`                | Number of gaussians (splat nodes) |
| `centroid`          | `tuple[float, float, float]` | Node centroid             |
| `world_transform`   | `tuple`              | World-space 4x4 transform        |
| `splat_data()`      | `SplatData or None`  | Splat data for this node (None if not a splat) |
| `point_cloud()`     | `PointCloud or None` | Point cloud data (None if not a point cloud) |
| `cropbox()`         | `CropBox or None`    | Crop box data                     |
| `ellipsoid()`       | `Ellipsoid or None`  | Ellipsoid data                    |

### Selection

| Function                              | Returns          | Description                       |
|---------------------------------------|------------------|-----------------------------------|
| `select_node(name)`                   | `None`           | Select node by name               |
| `deselect_all()`                      | `None`           | Clear selection                   |
| `has_selection()`                     | `bool`           | Any selection active              |
| `get_selected_node_name()`            | `str`            | First selected node name          |
| `get_selected_node_names()`           | `list[str]`      | All selected node names           |
| `can_transform_selection()`           | `bool`           | Selection is transformable        |
| `get_selected_node_transform()`       | `list[float]`    | 16 floats, column-major           |
| `set_selected_node_transform(matrix)` | `None`           | Set transform                     |
| `get_selection_center()`              | `list[float]`    | Local space center                |
| `get_selection_world_center()`        | `list[float]`    | World space center                |
| `capture_selection_transforms()`      | `dict`           | Snapshot for undo                 |

### Scene Shortcuts

Module-level shortcuts for common scene operations (equivalent to `Scene` object methods):

| Function | Returns | Description |
|----------|---------|-------------|
| `set_node_visibility(name, visible)` | `None` | Toggle node visibility |
| `remove_node(name, keep_children=False)` | `None` | Remove node |
| `reparent_node(name, new_parent)` | `None` | Reparent node |
| `rename_node(old_name, new_name)` | `None` | Rename node |
| `add_group(name, parent="")` | `None` | Add group node |
| `get_num_gaussians()` | `int` | Total gaussian count |

### Gaussian-Level Selection (on Scene object)

| Method                          | Returns     | Description                  |
|---------------------------------|-------------|------------------------------|
| `set_selection_mask(mask)`      | `None`      | Apply bool tensor mask       |
| `clear_selection()`             | `None`      | Clear gaussian selection     |
| `has_selection()`               | `bool`      | Any gaussians selected       |
| `selection_mask`                | `Tensor`    | Property. Current mask       |
| `set_selection(indices)`        | `None`      | Select by index list         |

### Transforms

| Function                                    | Returns        | Description                      |
|---------------------------------------------|----------------|----------------------------------|
| `get_node_transform(name)`                  | `list[float]`  | 16 floats, column-major          |
| `set_node_transform(name, matrix)`          | `None`         | Set 4x4 transform                |
| `decompose_transform(matrix)`               | `dict`         | See keys below                   |
| `compose_transform(translation, euler_deg, scale)` | `list[float]` | Build 4x4 from components (Euler in degrees) |

`decompose_transform` returns a dict with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `translation` | `[x, y, z]` | Position |
| `rotation_quat` | `[x, y, z, w]` | Quaternion |
| `rotation_euler` | `[rx, ry, rz]` | Euler angles (radians) |
| `rotation_euler_deg` | `[rx, ry, rz]` | Euler angles (degrees) |
| `scale` | `[sx, sy, sz]` | Scale |

### Splat Data (combined_model() / node.splat_data())

Accessible via `scene.combined_model()` (all nodes merged) or `node.splat_data()` (per-node).

| Property/Method       | Returns      | Description                     |
|-----------------------|--------------|---------------------------------|
| `means_raw`           | `Tensor`     | [N, 3] positions (view)        |
| `sh0_raw`             | `Tensor`     | [N, 1, 3] base SH (view)       |
| `shN_raw`             | `Tensor`     | [N, K, 3] higher SH (view)     |
| `scaling_raw`         | `Tensor`     | [N, 3] log-space (view)        |
| `rotation_raw`        | `Tensor`     | [N, 4] quaternions (view)      |
| `opacity_raw`         | `Tensor`     | [N, 1] logit-space (view)      |
| `get_means()`         | `Tensor`     | Positions                       |
| `get_opacity()`       | `Tensor`     | [N] sigmoid applied             |
| `get_scaling()`       | `Tensor`     | Exp applied                     |
| `get_rotation()`      | `Tensor`     | Normalized quaternions          |
| `get_shs()`           | `Tensor`     | SH0 + SHN concatenated         |
| `num_points`          | `int`        | Gaussian count                  |
| `active_sh_degree`    | `int`        | Current SH degree               |
| `max_sh_degree`       | `int`        | Maximum SH degree               |
| `scene_scale`         | `float`      | Scene scale factor              |
| `soft_delete(mask)`   | `Tensor`     | Mark for deletion, returns prev state |
| `undelete(mask)`      | `None`       | Restore deleted gaussians       |
| `apply_deleted()`     | `int`        | Permanently remove, returns count|
| `clear_deleted()`     | `None`       | Clear deletion mask             |
| `deleted`             | `Tensor`     | Property. [N] bool deletion mask|
| `has_deleted_mask()`  | `bool`       | Whether deletion mask exists    |
| `visible_count()`     | `int`        | Number of non-deleted gaussians |

> After calling `soft_delete()`, `undelete()`, or `clear_deleted()`, call `scene.notify_changed()` to update the viewport.

### Training Control

| Function                    | Returns          | Description                   |
|-----------------------------|------------------|-------------------------------|
| `start_training()`         | `None`           | Start training                |
| `pause_training()`         | `None`           | Pause                         |
| `resume_training()`        | `None`           | Resume                        |
| `stop_training()`          | `None`           | Stop                          |
| `reset_training()`         | `None`           | Reset to iteration 0          |
| `save_checkpoint()`        | `None`           | Save checkpoint               |
| `switch_to_edit_mode()`    | `None`           | Enter edit mode               |
| `has_trainer()`            | `bool`           | Trainer loaded                |
| `trainer_state()`          | `str`            | State string                  |
| `finish_reason()`          | `str or None`    | Why training ended            |
| `trainer_error()`          | `str or None`    | Error message                 |
| `context()`                | `Context`        | Training context snapshot     |
| `optimization_params()`    | `OptimizationParams` | Training parameters      |
| `dataset_params()`         | `DatasetParams`  | Dataset parameters            |
| `loss_buffer()`            | `list[float]`    | Loss history                  |
| `load_checkpoint_for_training(checkpoint_path, dataset_path, output_path)` | `None` | Load checkpoint for training |

### Training Status

| Function | Returns | Description |
|----------|---------|-------------|
| `trainer_elapsed_seconds()` | `float` | Elapsed training time |
| `trainer_eta_seconds()` | `float` | Estimated remaining time (-1 if unavailable) |
| `trainer_strategy_type()` | `str` | Strategy type (mcmc, default, etc.) |
| `trainer_is_gut_enabled()` | `bool` | GUT enabled |
| `trainer_max_gaussians()` | `int` | Max gaussians |
| `trainer_num_splats()` | `int` | Current splat count |
| `trainer_current_iteration()` | `int` | Current iteration |
| `trainer_total_iterations()` | `int` | Total iterations |
| `trainer_current_loss()` | `float` | Current loss |

### Training Hooks

| Decorator                    | Description                         |
|------------------------------|-------------------------------------|
| `@lf.on_training_start`     | Called when training starts          |
| `@lf.on_iteration_start`    | Called at start of each iteration    |
| `@lf.on_pre_optimizer_step` | Called before optimizer step         |
| `@lf.on_post_step`          | Called after each step               |
| `@lf.on_training_end`       | Called when training ends            |

### Rendering

| Function                                              | Returns          | Description                |
|-------------------------------------------------------|------------------|----------------------------|
| `get_current_view()`                                  | `ViewInfo`       | Current camera view        |
| `get_viewport_render()`                               | `ViewportRender` | Current viewport image     |
| `capture_viewport()`                                  | `ViewportRender` | Capture for async use      |
| `render_view(rotation, translation, w, h, fov=60, bg=None)` | `Tensor`  | Render from camera         |
| `compute_screen_positions(rotation, translation, w, h, fov=60)` | `Tensor` | [N, 2] screen positions |
| `get_render_settings()`                               | `RenderSettings` | Current render settings    |
| `get_render_mode()` / `set_render_mode(mode)`         | `RenderMode`     | Render mode                |

### Viewport Control

| Function                        | Returns | Description              |
|---------------------------------|---------|--------------------------|
| `reset_camera()`               | `None`  | Reset camera             |
| `toggle_fullscreen()`          | `None`  | Toggle fullscreen        |
| `is_fullscreen()`              | `bool`  | Fullscreen state         |
| `toggle_ui()`                  | `None`  | Toggle UI visibility     |
| `set_orthographic(ortho)`      | `None`  | Set projection mode      |
| `is_orthographic()`            | `bool`  | Orthographic state       |

### Export

```python
lf.export_scene(
    format: int,             # 0=PLY, 1=SOG, 2=SPZ, 3=HTML
    path: str,
    node_names: list[str],
    sh_degree: int,
)
lf.save_config_file(path: str)
```

### Logging

| Function           | Description       |
|--------------------|-------------------|
| `lf.log.info(msg)` | Info level        |
| `lf.log.warn(msg)` | Warning level     |
| `lf.log.error(msg)`| Error level       |
| `lf.log.debug(msg)`| Debug level       |

### Undo

```python
lf.undo.push(name: str, undo: Callable, redo: Callable, validate: Callable | None = None)
```

### UI Functions

| Function                                    | Returns          | Description                |
|---------------------------------------------|------------------|----------------------------|
| `lf.ui.tr(key)`                             | `str`            | Translate string           |
| `lf.ui.theme()`                             | `Theme`          | Current theme              |
| `lf.ui.context()`                           | `AppContext`     | App context                |
| `lf.ui.request_redraw()`                    | `None`           | Request UI redraw          |
| `lf.ui.set_language(lang_code)`             | `None`           | Set UI language            |
| `lf.ui.get_current_language()`              | `str`            | Active language code       |
| `lf.ui.get_languages()`                     | `list[tuple[str, str]]` | Available languages  |
| `lf.ui.set_theme(name)`                     | `None`           | Theme switch (`dark`/`light`) |
| `lf.ui.get_theme()`                         | `str`            | Active theme name          |
| `lf.ui.set_panel_enabled(panel_id, enabled)`  | `None`           | Toggle panel by id         |
| `lf.ui.is_panel_enabled(panel_id)`            | `bool`           | Panel enabled state        |
| `lf.ui.get_panel_names(space=lf.ui.PanelSpace.FLOATING)` | `list[str]` | Panel ids for a space |
| `lf.ui.get_panel(panel_id)`                   | `lf.ui.PanelInfo \| None`   | Typed panel info |
| `lf.ui.get_main_panel_tabs()`                 | `list[lf.ui.PanelSummary]` | Typed summaries for main-panel tabs |
| `lf.ui.set_panel_label(panel_id, label)`      | `bool`           | Change panel display name  |
| `lf.ui.set_panel_order(panel_id, order)`      | `bool`           | Change panel sort order    |
| `lf.ui.set_panel_space(panel_id, space)`      | `bool`           | Move panel to a different space (`lf.ui.PanelSpace`) |
| `lf.ui.set_panel_parent(panel_id, parent)`    | `bool`           | Embed panel inside a tab as collapsible section |
| `lf.ui.ops.invoke(op_id, **kwargs)`         | `OperatorReturnValue` | Invoke operator       |
| `lf.ui.ops.poll(op_id)`                     | `bool`           | Operator poll              |
| `lf.ui.ops.cancel_modal()`                  | `None`           | Cancel modal operator      |
| `lf.ui.get_active_tool()`                   | `str`            | Active tool ID             |
| `lf.ui.get_active_submode()`                | `str`            | Active submode             |
| `lf.ui.set_selection_mode(mode)`            | `None`           | Set selection submode      |
| `lf.ui.get_transform_space()`               | `int`            | Transform space enum index |
| `lf.ui.set_transform_space(space)`          | `None`           | Set transform space index  |
| `lf.ui.get_pivot_mode()` / `set_pivot_mode(mode)` | `int`      | Pivot mode enum index      |
| `lf.ui.get_fps()`                           | `float`          | Current FPS                |
| `lf.ui.get_gpu_memory()`                    | `(int, int, int)` | (process_used, total_used, total) bytes |
| `lf.ui.get_git_commit()`                    | `str`            | Git commit hash            |

### File Dialogs

| Function                                    | Returns          |
|---------------------------------------------|------------------|
| `lf.ui.open_image_dialog(start_dir='')`     | `str`            |
| `lf.ui.open_folder_dialog(title='Select Folder', start_dir='')` | `str` |
| `lf.ui.open_dataset_folder_dialog()`        | `str`            |
| `lf.ui.open_ply_file_dialog(start_dir='')`  | `str`            |
| `lf.ui.open_mesh_file_dialog(start_dir='')` | `str`            |
| `lf.ui.open_checkpoint_file_dialog()`       | `str`            |
| `lf.ui.open_json_file_dialog()`             | `str`            |
| `lf.ui.open_video_file_dialog()`            | `str`            |
| `lf.ui.save_json_file_dialog(default_name='config.json')` | `str` |
| `lf.ui.save_ply_file_dialog(default_name='export.ply')`   | `str` |
| `lf.ui.save_sog_file_dialog(default_name='export.sog')`   | `str` |
| `lf.ui.save_spz_file_dialog(default_name='export.spz')`   | `str` |
| `lf.ui.save_html_file_dialog(default_name='viewer.html')` | `str` |

### UI Hooks

Inject UI into existing panels at predefined hook points. Callbacks receive a `layout` object.

| Function | Description |
|---|---|
| `lf.ui.add_hook(panel, section, callback, position="append")` | Register a hook. `position`: `"prepend"` or `"append"` |
| `lf.ui.remove_hook(panel, section, callback)` | Remove a specific hook callback |
| `lf.ui.clear_hooks(panel, section="")` | Clear hooks for panel/section (or all sections if empty) |
| `lf.ui.clear_all_hooks()` | Clear all registered hooks |
| `lf.ui.get_hook_points()` | List all registered hook point keys |
| `lf.ui.invoke_hooks(panel, section, prepend=False)` | Invoke hooks (`prepend=True` for prepend, `False` for append) |
| `@lf.ui.hook(panel, section, position="append")` | Decorator form of `add_hook` |

Hook points are runtime-defined. Query them with `lf.ui.get_hook_points()` instead of hard-coding.

### Tensor API

```python
import lichtfeld as lf
t = lf.Tensor
```

The tables below list the most-used tensor APIs. For the full bound surface, see `src/python/stubs/lichtfeld/__init__.pyi`.

**Creation:**

| Function                                    | Returns  | Description              |
|---------------------------------------------|----------|--------------------------|
| `t.zeros(shape, device='cuda', dtype='float32')` | `Tensor` | Zero-filled tensor   |
| `t.ones(shape, device, dtype)`              | `Tensor` | Ones tensor              |
| `t.full(shape, value, device, dtype)`       | `Tensor` | Constant-filled tensor   |
| `t.eye(n, device, dtype)`                   | `Tensor` | Identity matrix          |
| `t.arange(start, end, step, device, dtype)` | `Tensor` | Range tensor             |
| `t.linspace(start, end, steps, device, dtype)` | `Tensor` | Linear space          |
| `t.rand(shape, device, dtype)`              | `Tensor` | Uniform random [0, 1)   |
| `t.randn(shape, device, dtype)`             | `Tensor` | Normal random            |
| `t.empty(shape, device, dtype)`             | `Tensor` | Uninitialized tensor     |
| `t.randint(low, high, shape, device)`       | `Tensor` | Random integers          |
| `t.from_numpy(arr, copy=True)`              | `Tensor` | From NumPy array         |
| `t.cat(tensors, dim=0)`                     | `Tensor` | Concatenate              |
| `t.stack(tensors, dim=0)`                   | `Tensor` | Stack                    |
| `t.where(condition, x, y)`                  | `Tensor` | Conditional select       |

**Properties:**

| Property         | Type    | Description              |
|------------------|---------|--------------------------|
| `.shape`         | `tuple` | Tensor dimensions        |
| `.ndim`          | `int`   | Number of dimensions     |
| `.numel`         | `int`   | Total elements           |
| `.device`        | `str`   | `'cpu'` or `'cuda'`     |
| `.dtype`         | `str`   | Data type string         |
| `.is_contiguous` | `bool`  | Memory contiguous        |
| `.is_cuda`       | `bool`  | On GPU                   |

**Methods:**

| Method                              | Returns  | Description              |
|-------------------------------------|----------|--------------------------|
| `.clone()`                          | `Tensor` | Deep copy                |
| `.cpu()` / `.cuda()`               | `Tensor` | Move device              |
| `.contiguous()`                     | `Tensor` | Make contiguous          |
| `.sync()`                           | `None`   | CUDA synchronize         |
| `.numpy(copy=True)`                 | `ndarray`| Convert to NumPy         |
| `.to(dtype)`                        | `Tensor` | Convert dtype            |
| `.size(dim)`                        | `int`    | Size at dimension        |
| `.item()`                           | `scalar` | Extract scalar           |
| `.sum(dim=None, keepdim=False)`     | `Tensor` | Reduce sum               |
| `.mean(dim=None, keepdim=False)`    | `Tensor` | Reduce mean              |
| `.max(dim=None, keepdim=False)`     | `Tensor` | Reduce max               |
| `.min(dim=None, keepdim=False)`     | `Tensor` | Reduce min               |
| `.reshape(shape)`                   | `Tensor` | Reshape                  |
| `.view(shape)`                      | `Tensor` | View reshape             |
| `.squeeze(dim=None)`                | `Tensor` | Remove size-1 dims       |
| `.unsqueeze(dim)`                   | `Tensor` | Add size-1 dim           |
| `.transpose(dim0, dim1)`            | `Tensor` | Swap dimensions          |
| `.permute(dims)`                    | `Tensor` | Reorder dimensions       |
| `.flatten(start=0, end=-1)`         | `Tensor` | Flatten range            |
| `.expand(sizes)`                    | `Tensor` | Broadcast view           |
| `.repeat(repeats)`                  | `Tensor` | Tile tensor              |
| `.prod()`, `.std()`, `.var()`      | `Tensor` | Additional reductions    |
| `.argmax()`, `.argmin()`            | `Tensor` | Index reductions         |
| `.all()`, `.any()`                  | `Tensor` | Logical reductions       |
| `.matmul()`, `.mm()`, `.bmm()`      | `Tensor` | Matrix products          |
| `.masked_select()`, `.masked_fill()`| `Tensor` | Masked operations        |
| `.zeros_like()`, `.ones_like()` etc.| `Tensor` | Like-constructors        |
| `.from_dlpack()` / `.__dlpack__()`  | `Tensor` | DLPack interop           |

**Operators:** `+`, `-`, `*`, `/`, `**`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `[]` (indexing/slicing)

### Application

| Function             | Description              |
|----------------------|--------------------------|
| `lf.request_exit()`  | Exit with confirmation   |
| `lf.force_exit()`    | Immediate exit           |
| `lf.run(path)`       | Execute Python script    |
| `lf.on_frame(cb)`    | Per-frame callback       |
| `lf.stop_animation()`| Clear frame callback     |
| `lf.mat4(rows)`      | Create 4x4 matrix        |
| `lf.help()`          | Show help                |

---

## pyproject.toml Schema

```toml
[project]
name = ""                    # string, required - Unique plugin identifier
version = ""                 # string, required - Semantic version
description = ""             # string, required
authors = []                 # list[{name, email}], optional - PEP 621 authors
dependencies = []            # list[string], optional - Python packages (PEP 508)

[tool.lichtfeld]
hot_reload = true            # bool, required
entry_point = "__init__"     # string, optional - Module to load (default: __init__)
plugin_api = ">=1,<2"        # string, required - Supported plugin API range (PEP 440)
lichtfeld_version = ">=0.4.2"  # string, required - Supported host app/runtime range (PEP 440)
required_features = []       # list[string], required - Optional host features this plugin needs
author = ""                  # string, optional - Author fallback (if no [project].authors)
```

v1 is strict. Legacy `min_lichtfeld_version` / `max_lichtfeld_version` fields are removed and rejected.

---

## Icon System

```python
from lfs_plugins.icon_manager import get_icon, get_ui_icon, get_scene_icon, get_plugin_icon
```

| Function                                    | Returns | Description                              |
|---------------------------------------------|---------|------------------------------------------|
| `get_icon(name)`                            | `int`   | Load `assets/icon/{name}.png`            |
| `get_ui_icon(name)`                         | `int`   | Load `assets/icon/{name}` (include ext)  |
| `get_scene_icon(name)`                      | `int`   | Load `assets/icon/scene/{name}.png`      |
| `get_plugin_icon(name, plugin_path, plugin_name)` | `int` | Load `{plugin_path}/icons/{name}.png` with fallback |

All return OpenGL texture ID (0 on failure). Icons are cached by C++.

Direct loading:
```python
import lichtfeld as lf
texture_id = lf.load_icon(name)
lf.free_icon(texture_id)
```

---

## Errors

Plugin errors are captured and accessible via the plugin manager:

```python
import lichtfeld as lf

state = lf.plugins.get_state("my_plugin")   # PluginState enum
error = lf.plugins.get_error("my_plugin")   # Error message string
tb = lf.plugins.get_traceback("my_plugin")  # Full traceback string
```

### PluginState values

| State        | Description                    |
|--------------|--------------------------------|
| `UNLOADED`   | Plugin is not loaded           |
| `INSTALLING` | Plugin is being installed      |
| `LOADING`    | Plugin is loading              |
| `ACTIVE`     | Plugin is running              |
| `ERROR`      | Plugin failed to load/run      |
| `DISABLED`   | Plugin is manually disabled    |
# Plugin Developer Guide

LichtFeld Studio plugins extend the application with panels, operators, tools, signals, and capabilities. Plugins live in `~/.lichtfeld/plugins/` and are just Python packages with a small manifest and entrypoint.

## Learning path

Read the examples in this order:

| Step | Goal | Example |
|---|---|---|
| 1 | Pure immediate-mode panel with `draw(ui)` only | [`examples/01_draw_only.py`](examples/01_draw_only.py) |
| 2 | Add shell, styling, and periodic updates without rewriting `draw(ui)` | [`examples/02_status_bar_mixed.py`](examples/02_status_bar_mixed.py) |
| 3 | Build a full hybrid panel with template, RCSS, data model, DOM hooks, and embedded `draw(ui)` | [`examples/03_hybrid_plugin/`](examples/03_hybrid_plugin/) |
| 4 | Explore focused feature demos | [`examples/README.md`](examples/README.md) |
| 5 | See an end-to-end multi-file plugin | [`examples/full_plugin/`](examples/full_plugin/) |

The key idea is that `lf.ui.Panel` is one public base class that scales from the smallest `draw(ui)` panel to full retained/hybrid UI. You do not need to switch APIs or rewrite the panel body when you add advanced features.

## Quick start

### Plugin directory structure

```text
~/.lichtfeld/plugins/my_plugin/
├── pyproject.toml       # Plugin manifest (required)
├── __init__.py          # Entry point with on_load/on_unload (required)
├── panels/
│   ├── __init__.py
│   ├── main_panel.py
│   ├── main_panel.rml   # Scaffolded for v1; optional to customize
│   └── main_panel.rcss  # Scaffolded sibling stylesheet
├── operators/           # Optional
│   └── my_operator.py
└── icons/               # Optional PNG icons for custom tools
    └── my_icon.png
```

### Scaffold with CLI or Python

Create a plugin from the command line when you also want a venv and editor config:

```bash
LichtFeld-Studio plugin create my_plugin
LichtFeld-Studio plugin check my_plugin
LichtFeld-Studio plugin list
```

Create a plugin from Python when you only want the source package:

```python
import lichtfeld as lf

path = lf.plugins.create("my_plugin")
print(path)
```

Important scaffold behavior:

- `lf.plugins.create()` writes `pyproject.toml`, `__init__.py`, `panels/__init__.py`, `panels/main_panel.py`, `panels/main_panel.rml`, and `panels/main_panel.rcss`.
- `LichtFeld-Studio plugin create` writes the same source files and also adds `.venv/`, `.vscode/`, and `pyrightconfig.json`.
- The scaffold is hybrid-ready, but you can ignore the retained files until you actually need custom DOM or standalone RCSS.

That is intentional. Most plugins should still start by editing `draw(ui)` in `main_panel.py`; v1 just removes the later migration work when you decide to add a custom template.

### `pyproject.toml`

Every plugin needs `[project]` metadata and a `[tool.lichtfeld]` section:

```toml
[project]
name = "my_plugin"
version = "0.1.0"
description = "What this plugin does"
authors = [{name = "Your Name"}]
dependencies = []

[tool.lichtfeld]
hot_reload = true
plugin_api = ">=1,<2"
lichtfeld_version = ">=0.4.2"
required_features = []
```

Notes:

- `name`, `version`, and `description` are required.
- `hot_reload` is required.
- `plugin_api`, `lichtfeld_version`, and `required_features` are required in v1.
- Legacy `min_lichtfeld_version` / `max_lichtfeld_version` fields are removed and rejected.
- Plugin-local Python dependencies go in `project.dependencies`.
- Inspect the current host contract from Python with `lf.PLUGIN_API_VERSION`, `lf.plugins.API_VERSION`, and `lf.plugins.FEATURES`.

### `__init__.py`

Your entrypoint must define `on_load()` and `on_unload()`:

```python
import lichtfeld as lf
from .panels.main_panel import MainPanel

_classes = [MainPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("my_plugin loaded")


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("my_plugin unloaded")
```

## Panels

Panels are the main UI surface for most plugins. The same `lf.ui.Panel` class supports both immediate-mode and retained/hybrid UI.

### Step 1: start with `draw(ui)`

This is the smallest useful panel:

```python
import lichtfeld as lf


class HelloPanel(lf.ui.Panel):
    id = "hello_world.main_panel"
    label = "Hello World"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 200

    def __init__(self):
        self._clicks = 0

    def draw(self, ui):
        ui.heading("Hello from my plugin")
        ui.text_disabled("This panel uses only draw(ui).")

        if ui.button_styled(f"Greet ({self._clicks})", "primary"):
            self._clicks += 1
            lf.log.info("Hello, LichtFeld!")
```

That alone is enough to ship a plugin panel. Keep state on `self`, render with `draw(ui)`, and register the class in `on_load()`.

See the full version in [`examples/01_draw_only.py`](examples/01_draw_only.py).

### Panel attributes

```python
import lichtfeld as lf


class MyPanel(lf.ui.Panel):
    id = "my_plugin.panel"
    label = "My Panel"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    parent = ""
    order = 100
    options = set()
    poll_dependencies = {
        lf.ui.PollDependency.SCENE,
        lf.ui.PollDependency.SELECTION,
        lf.ui.PollDependency.TRAINING,
    }
    size = None
    template = ""
    style = ""
    height_mode = lf.ui.PanelHeightMode.FILL
    update_interval_ms = 100

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def draw(self, ui):
        ui.label("Content here")
```

| Attribute | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | `module.qualname` | Unique panel identifier. Used for replacement, visibility, and API lookups. |
| `label` | `str` | `""` | Display name in the UI. Falls back to `id` when empty. |
| `space` | `lf.ui.PanelSpace` | `lf.ui.PanelSpace.MAIN_PANEL_TAB` | Where the panel appears when `parent` is empty. |
| `parent` | `str` | `""` | Parent panel id. When set, the panel embeds as a collapsible section and must not also override `space`. |
| `order` | `int` | `100` | Sort order within its space. Lower values appear earlier. |
| `options` | `set[lf.ui.PanelOption]` | `set()` | Panel options such as `lf.ui.PanelOption.DEFAULT_CLOSED` and `lf.ui.PanelOption.HIDE_HEADER`. |
| `poll_dependencies` | `set[lf.ui.PollDependency]` | `{SCENE, SELECTION, TRAINING}` | Which app-state changes should re-run `poll()`. |
| `size` | `tuple[float, float] \| None` | `None` | Initial width/height hint, mainly useful for floating panels. |
| `template` | `str \| os.PathLike[str]` | `""` | Optional retained RML template. Use an absolute path for plugin-local files. |
| `style` | `str` | `""` | Optional inline RCSS appended to the retained document. This is RCSS text, not a file path. |
| `height_mode` | `lf.ui.PanelHeightMode` | `lf.ui.PanelHeightMode.FILL` | `FILL` or `CONTENT` for retained panels. |
| `update_interval_ms` | `int` | `100` | Update cadence for retained/hybrid `on_update()` work. |

The panel API is strict in v1: use the enum values above, not string literals.

Panel definitions are validated eagerly. Invalid enum values, removed legacy fields, retained-only settings in `VIEWPORT_OVERLAY`, and conflicting fields such as `parent` plus explicit `space` raise `ValueError`, `TypeError`, or `AttributeError` during `lf.register_class()`.

### Step 2: add shell and retained behavior without rewriting `draw(ui)`

The unified API is designed for progressive disclosure. You can keep `draw(ui)` as your content source and opt into advanced features on the same class:

```python
import lichtfeld as lf


class StatusBarPanel(lf.ui.Panel):
    id = "my_plugin.status"
    label = "Build Up 2"
    space = lf.ui.PanelSpace.STATUS_BAR
    height_mode = lf.ui.PanelHeightMode.CONTENT
    update_interval_ms = 120
    style = """
body.status-bar-panel { padding: 0 12dp; }
#im-root .im-label { color: #f3c96d; font-weight: bold; }
"""

    def __init__(self):
        self._progress = 0.2

    def draw(self, ui):
        ui.label("STATUS")
        ui.progress_bar(self._progress, f"{int(self._progress * 100)}%")

    def on_update(self, doc):
        del doc
        self._progress = (self._progress + 0.02) % 1.0
        return True
```

What changes here:

- `style` adds inline RCSS.
- `height_mode` controls how the retained shell sizes itself.
- `on_update()` adds periodic behavior.
- `draw(ui)` still renders the actual content.

This is the normal upgrade path. You do not need to rewrite the panel as full DOM/RML just because you added styling or retained hooks.

See the full version in [`examples/02_status_bar_mixed.py`](examples/02_status_bar_mixed.py).

### Retained shells and template resolution

When a panel uses retained features, LichtFeld chooses a shell automatically if `template` is empty:

| Space | Default retained shell |
|---|---|
| `FLOATING` | `rmlui/floating_window.rml` |
| `STATUS_BAR` | `rmlui/status_bar_panel.rml` |
| Other retained panel spaces | `rmlui/docked_panel.rml` |

Built-in aliases:

- `builtin:docked-panel`
- `builtin:floating-window`
- `builtin:status-bar`

For plugin-local templates, prefer absolute paths:

```python
from pathlib import Path

template = str(Path(__file__).resolve().with_name("main_panel.rml"))
```

When a template file exists at `main_panel.rml`, LichtFeld automatically looks for a sibling `main_panel.rcss` file and loads it as the base stylesheet for that document.

### Which styling path should you use?

| Goal | Best tool | Extra files |
|---|---|---|
| Start simple and ship quickly | `draw(ui)` plus built-in widgets and sub-layouts | None |
| Tweak spacing, colors, or typography on a retained shell | `style` with inline RCSS | None |
| Own the DOM structure and stylesheet | `template` plus sibling `.rml` and `.rcss` | `main_panel.rml`, `main_panel.rcss` |

Use that ladder in order. The scaffold still starts with immediate-mode content on the first row even though the retained shell files are already present.

### Step 3: go full hybrid

Use a custom template when you want retained DOM structure, data binding, or direct event listeners, but still keep an embedded immediate-mode area when that is convenient.

```python
from pathlib import Path
import lichtfeld as lf

MODEL_NAME = "my_plugin_hybrid"


class HybridPanel(lf.ui.Panel):
    id = "my_plugin.hybrid"
    label = "Hybrid"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT

    def draw(self, ui):
        ui.text_disabled("This block is rendered into #im-root.")

    def on_bind_model(self, ctx):
        model = ctx.create_data_model(MODEL_NAME)
        if model is None:
            return
        model.bind_func("title", lambda: "Hybrid Panel")
        self._handle = model.get_handle()

    def on_mount(self, doc):
        header = doc.get_element_by_id("header")
        if header:
            header.add_event_listener("click", lambda _ev: lf.log.info("Header clicked"))

    def on_update(self, doc):
        del doc
        if getattr(self, "_handle", None):
            self._handle.dirty_all()
```

Key retained hooks:

- `on_bind_model(ctx)`: create and bind a retained data model before the document loads.
- `on_mount(doc)`: wire DOM listeners or build dynamic DOM content after the document mounts.
- `on_unmount(doc)`: clean up document-local state.
- `on_update(doc)`: periodic updates while the panel is visible. Return `True` to mark content dirty.
- `on_scene_changed(doc)`: respond to active scene generation changes.

To mix retained and immediate content, include `<div id="im-root"></div>` somewhere in your template. `draw(ui)` will render into that node.

See the complete multi-file example in [`examples/03_hybrid_plugin/`](examples/03_hybrid_plugin/).

### Panel spaces

| Space | Description |
|---|---|
| `MAIN_PANEL_TAB` | Own tab in the right panel. Default for plugin panels. |
| `SIDE_PANEL` | Right sidebar panel. |
| `VIEWPORT_OVERLAY` | Drawn over the 3D viewport. |
| `SCENE_HEADER` | Header area above the scene tree. |
| `FLOATING` | Free-floating window. |
| `STATUS_BAR` | Bottom status bar. |

### Embedding in an existing tab

Use `parent` to place your panel inside a built-in tab as a collapsible section:

```python
class MyAnalysis(lf.ui.Panel):
    label = "My Analysis"
    parent = "lfs.rendering"
    order = 200

    def draw(self, ui):
        ui.label("Analysis results here")
```

Common parent ids:

| `parent` value | Effect |
|---|---|
| `"lfs.rendering"` | Collapsible section inside Rendering |
| `"lfs.training"` | Collapsible section inside Training |

### Register and unregister

```python
import lichtfeld as lf

lf.register_class(MyPanel)
lf.unregister_class(MyPanel)
```

### Panel replacement

Registering a panel with the same `id` as an existing panel replaces it. This is how plugins override built-in panels:

```python
import lichtfeld as lf


class MyTrainingPanel(lf.ui.Panel):
    id = "lfs.training"
    label = "Training"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 20

    def draw(self, ui):
        ui.label("Custom training controls")
```

Third-party plugins load after built-ins, so the replacement takes effect automatically while keeping the same slot in the UI.

### Panel management API

```python
import lichtfeld as lf

lf.ui.set_panel_enabled("my_plugin.panel", False)
lf.ui.is_panel_enabled("my_plugin.panel")

lf.ui.get_panel("my_plugin.panel")
lf.ui.set_panel_label("my_plugin.panel", "New Name")
lf.ui.set_panel_order("my_plugin.panel", 50)
lf.ui.set_panel_space("my_plugin.panel", lf.ui.PanelSpace.FLOATING)
lf.ui.set_panel_parent("my_plugin.panel", "lfs.rendering")
lf.ui.get_panel_names(lf.ui.PanelSpace.MAIN_PANEL_TAB)
```

`lf.ui.get_panel()` returns a typed `lf.ui.PanelInfo | None`, and `lf.ui.get_main_panel_tabs()` returns `list[lf.ui.PanelSummary]`.

### Layout composition

The `ui` object passed to `draw()` exposes a large widget/layout API. Start with direct calls, then use sub-layouts when structure matters:

```python
def draw(self, ui):
    with ui.row() as row:
        row.button("Action A")
        row.button("Action B")

    with ui.column() as col:
        col.label("Top")
        col.label("Bottom")

    with ui.box() as box:
        box.heading("Settings")
        box.prop(self, "opacity")

    with ui.split(0.3) as split:
        split.label("Name")
        split.prop(self, "name")

    with ui.grid_flow(columns=3) as grid:
        for item in items:
            grid.button(item.name)
```

See [examples/README.md](examples/README.md) for the recommended progression through the example files.

### Example: viewport overlay

```python
import lichtfeld as lf
from lfs_plugins.ui.state import AppState


class StatsOverlay(lf.ui.Panel):
    label = "Stats"
    space = lf.ui.PanelSpace.VIEWPORT_OVERLAY
    order = 10

    @classmethod
    def poll(cls, context) -> bool:
        return AppState.has_scene.value

    def draw(self, ui):
        n = AppState.num_gaussians.value
        ui.draw_text(10, 10, f"Gaussians: {n:,}", (1.0, 1.0, 1.0, 0.8))
```

### Displaying GPU tensors

Use `image_tensor` to render a CUDA tensor directly in a panel with no manual texture management:

```python
class PreviewPanel(lf.ui.Panel):
    label = "Preview"
    space = lf.ui.PanelSpace.FLOATING

    def draw(self, ui):
        tensor = lf.Tensor.rand([256, 256, 3], device="cuda")
        ui.image_tensor("my_preview", tensor, (256, 256))
```

The `label` argument (`"my_preview"`) caches the underlying GL texture between frames. Passing a tensor with a different resolution automatically recreates the texture. The tensor must be `[H, W, 3]` (RGB) or `[H, W, 4]` (RGBA). CPU tensors and integer dtypes are converted automatically.

For advanced use cases, use `DynamicTexture`:

```python
class AdvancedPanel(lf.ui.Panel):
    label = "Advanced"
    space = lf.ui.PanelSpace.FLOATING

    def __init__(self):
        self.tex = lf.ui.DynamicTexture()

    def draw(self, ui):
        self.tex.update(my_tensor)
        ui.image_texture(self.tex, (256, 256))
```

See the [DynamicTexture API reference](api-reference.md#dynamictexture) for all properties and methods.

---

## UI Hooks

Hooks let you inject UI into existing panels without replacing them. A hook callback receives a `layout` object and draws into the host panel at a predefined hook point.

### Hook pattern

```python
import lichtfeld as lf


class MyHookPanel:
    def draw(self, layout):
        if not layout.collapsing_header("My Section", default_open=True):
            return
        layout.label("Injected into the rendering panel")


_instance = None


def _draw_hook(layout):
    global _instance
    if _instance is None:
        _instance = MyHookPanel()
    _instance.draw(layout)


def register():
    lf.ui.add_hook("rendering", "selection_groups", _draw_hook, "append")


def unregister():
    lf.ui.remove_hook("rendering", "selection_groups", _draw_hook)
```

The `position` argument controls whether the hook draws before (`"prepend"`) or after (`"append"`) the native content at that hook point.

### Available hook points

| Panel | Section | Description |
|---|---|---|
| `"rendering"` | `"selection_groups"` | Rendering panel, between settings and tools |

### Decorator form

```python
@lf.ui.hook("rendering", "selection_groups", "append")
def my_hook(layout):
    layout.label("Hello from hook")
```

---

## Operators

Operators are actions that can be invoked by buttons, menus, or keyboard shortcuts. They extend `PropertyGroup`, so they can have typed properties.

### Operator base class

```python
from lfs_plugins.types import Operator, Event

class MyOperator(Operator):
    label = "My Action"
    description = "What this operator does"
    options = set()          # e.g. {'UNDO', 'BLOCKING'}

    @classmethod
    def poll(cls, context) -> bool:
        """Return False to disable the operator."""
        return True

    def invoke(self, context, event: Event) -> set:
        """Called when operator is first triggered. Can start modal."""
        return self.execute(context)

    def execute(self, context) -> set:
        """Synchronous execution."""
        return {"FINISHED"}

    def modal(self, context, event: Event) -> set:
        """Handle events during modal execution."""
        return {"FINISHED"}

    def cancel(self, context):
        """Called when the operator is cancelled."""
        pass
```

### Return sets

| Value             | Meaning                              |
|-------------------|--------------------------------------|
| `{"FINISHED"}`    | Operator completed successfully      |
| `{"CANCELLED"}`   | Operator was cancelled               |
| `{"RUNNING_MODAL"}` | Operator is running in modal mode |
| `{"PASS_THROUGH"}`  | Pass event to other handlers       |

Operators can also return a dict: `{"status": "FINISHED", "result": data}`.

### Event object

The `Event` object is passed to `invoke()` and `modal()`:

| Attribute        | Type    | Description                              |
|------------------|---------|------------------------------------------|
| `type`           | `str`   | `'MOUSEMOVE'`, `'LEFTMOUSE'`, `'KEY_A'`-`'KEY_Z'`, `'ESC'`, `'RET'`, `'SPACE'`, `'WHEELUPMOUSE'`, `'WHEELDOWNMOUSE'`, etc. |
| `value`          | `str`   | `'PRESS'`, `'RELEASE'`, `'NOTHING'`      |
| `mouse_x`        | `float` | Mouse X (viewport coords)               |
| `mouse_y`        | `float` | Mouse Y (viewport coords)               |
| `mouse_region_x` | `float` | Mouse X relative to region               |
| `mouse_region_y` | `float` | Mouse Y relative to region               |
| `delta_x`        | `float` | Mouse delta X                            |
| `delta_y`        | `float` | Mouse delta Y                            |
| `scroll_x`       | `float` | Scroll X offset                          |
| `scroll_y`       | `float` | Scroll Y offset                          |
| `shift`          | `bool`  | Shift held                               |
| `ctrl`           | `bool`  | Ctrl held                                |
| `alt`            | `bool`  | Alt held                                 |
| `pressure`       | `float` | Tablet pressure (1.0 for mouse)          |
| `over_gui`       | `bool`  | True if mouse is over a GUI element      |
| `key_code`       | `int`   | Key code (see `key_codes.hpp`)           |

### Example: simple execute-only operator

```python
import lichtfeld as lf
from lfs_plugins.types import Operator
from lfs_plugins.props import FloatProperty

class ResetOpacity(Operator):
    label = "Reset Opacity"
    description = "Set opacity of all gaussians to a given value"

    target_opacity: float = FloatProperty(default=1.0, min=0.0, max=1.0)

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()

    def execute(self, context) -> set:
        scene = lf.get_scene()
        model = scene.combined_model()
        n = model.num_points
        mask = lf.Tensor.ones([n, 1], device="cuda")
        scaled = mask * self.target_opacity
        # Apply to opacity (working in logit space requires inverse sigmoid)
        lf.log.info(f"Reset {n} gaussians to opacity {self.target_opacity}")
        return {"FINISHED"}
```

### Example: modal operator (interactive tool)

```python
import lichtfeld as lf
from lfs_plugins.types import Operator, Event

class MeasureTool(Operator):
    label = "Measure Distance"
    description = "Click two points to measure distance"
    options = {"UNDO"}

    def __init__(self):
        super().__init__()
        self.start_pos = None

    def invoke(self, context, event: Event) -> set:
        self.start_pos = None
        lf.log.info("Click first point...")
        return {"RUNNING_MODAL"}

    def modal(self, context, event: Event) -> set:
        if event.type == "ESC":
            lf.log.info("Measurement cancelled")
            return {"CANCELLED"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            pos = (event.mouse_x, event.mouse_y)
            if self.start_pos is None:
                self.start_pos = pos
                lf.log.info("Click second point...")
                return {"RUNNING_MODAL"}
            else:
                dx = pos[0] - self.start_pos[0]
                dy = pos[1] - self.start_pos[1]
                dist = (dx * dx + dy * dy) ** 0.5
                lf.log.info(f"Distance: {dist:.2f} pixels")
                return {"FINISHED"}

        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self.start_pos = None
```

---

## Toolbar Tools

Tools appear in the viewport toolbar and can have submodes and pivot modes.

### ToolDef dataclass

```python
from lfs_plugins.tool_defs.definition import ToolDef, SubmodeDef, PivotModeDef

tool = ToolDef(
    id="my_plugin.my_tool",         # Unique identifier
    label="My Tool",                # Display label
    icon="star",                    # Icon name
    group="utility",                # "select", "transform", "paint", "utility"
    order=200,                      # Sort order within group
    description="Tool tooltip",     # Tooltip
    shortcut="",                    # Keyboard shortcut
    gizmo="",                       # "translate", "rotate", "scale", or ""
    operator="",                    # Operator to invoke on activation
    submodes=(),                    # Tuple of SubmodeDef
    pivot_modes=(),                 # Tuple of PivotModeDef
    poll=None,                      # Callable[[context], bool]
    plugin_name="my_plugin",        # For custom icon loading
    plugin_path="/path/to/plugin",  # For custom icon loading
)
```

### Register and unregister

```python
from lfs_plugins.tools import ToolRegistry

ToolRegistry.register_tool(tool)
ToolRegistry.unregister_tool("my_plugin.my_tool")
```

### Custom icons

Place PNG icons in your plugin's `icons/` folder. Reference them by name (without extension) and set `plugin_name` and `plugin_path` on the `ToolDef`.

### Example: custom tool with submodes

```python
from pathlib import Path
from lfs_plugins.tool_defs.definition import ToolDef, SubmodeDef, PivotModeDef
from lfs_plugins.tools import ToolRegistry

paint_tool = ToolDef(
    id="my_plugin.paint",
    label="Paint",
    icon="paint",
    group="paint",
    order=100,
    description="Paint gaussian attributes",
    submodes=(
        SubmodeDef("opacity", "Opacity", "opacity"),
        SubmodeDef("color", "Color", "color"),
        SubmodeDef("scale", "Scale", "scale"),
    ),
    pivot_modes=(
        PivotModeDef("center", "Selection Center", "circle-dot"),
        PivotModeDef("cursor", "3D Cursor", "crosshair"),
    ),
    poll=lambda ctx: ctx.has_scene,
    plugin_name="my_plugin",
    plugin_path=str(Path(__file__).parent),
)

ToolRegistry.register_tool(paint_tool)
```

---

## Properties

Properties provide typed, validated attributes for operators and property groups.

### Property types

| Type                | Default     | Key Parameters                            |
|---------------------|-------------|-------------------------------------------|
| `FloatProperty`     | `0.0`       | `min`, `max`, `step`, `precision`, `subtype` |
| `IntProperty`       | `0`         | `min`, `max`, `step`                      |
| `BoolProperty`      | `False`     |                                           |
| `StringProperty`    | `""`        | `maxlen`, `subtype`                       |
| `EnumProperty`      | first item  | `items=[(id, label, desc), ...]`          |
| `FloatVectorProperty` | `(0,0,0)` | `size`, `min`, `max`, `subtype`           |
| `IntVectorProperty` | `(0,0,0)`  | `size`, `min`, `max`                      |
| `TensorProperty`    | `None`      | `shape`, `dtype`, `device`                |
| `CollectionProperty`| `[]`        | `type=PropertyGroupSubclass`              |
| `PointerProperty`   | `None`      | `type=PropertyGroupSubclass`              |

All properties accept: `name`, `description`, `subtype`, `update` (callback).

### PropertyGroup base class

```python
from lfs_plugins.props import PropertyGroup, FloatProperty, StringProperty

class MaterialSettings(PropertyGroup):
    color = FloatVectorProperty(default=(1, 1, 1), size=3, subtype="COLOR")
    roughness = FloatProperty(default=0.5, min=0.0, max=1.0)
    name = StringProperty(default="Untitled", maxlen=64)

# Singleton access
settings = MaterialSettings.get_instance()
settings.roughness = 0.8
print(settings.roughness)  # 0.8 (validated and clamped)
```

### Subtypes

| Subtype        | Applies To       | Effect                           |
|----------------|------------------|----------------------------------|
| `COLOR`        | FloatVector      | Color picker widget              |
| `COLOR_GAMMA`  | FloatVector      | Color picker with gamma          |
| `FILE_PATH`    | String           | File picker widget               |
| `DIR_PATH`     | String           | Folder picker widget             |
| `FACTOR`       | Float            | 0-1 slider                       |
| `PERCENTAGE`   | Float            | 0-100 slider                     |
| `ANGLE`        | Float            | Radians, displayed as degrees    |
| `TRANSLATION`  | FloatVector      | 3D translation                   |
| `EULER`        | FloatVector      | Euler rotation angles            |
| `QUATERNION`   | FloatVector(4)   | Quaternion rotation              |
| `XYZ`          | FloatVector      | Generic XYZ values               |

### Example: settings group with typed properties

```python
from lfs_plugins.props import (
    PropertyGroup, FloatProperty, IntProperty, BoolProperty,
    StringProperty, EnumProperty, FloatVectorProperty, TensorProperty,
)

class TrainingSettings(PropertyGroup):
    learning_rate = FloatProperty(
        default=0.001, min=0.0001, max=0.1,
        name="Learning Rate",
        description="Base learning rate for optimization",
    )
    max_iterations = IntProperty(default=30000, min=1000, max=100000)
    use_ssim = BoolProperty(default=True, name="Use SSIM Loss")
    output_path = StringProperty(default="output", subtype="DIR_PATH")
    strategy = EnumProperty(items=[
        ("mcmc", "MCMC", "Markov Chain Monte Carlo strategy"),
        ("default", "Default", "Default densification strategy"),
    ])
    background_color = FloatVectorProperty(
        default=(0.0, 0.0, 0.0), size=3, subtype="COLOR"
    )
    custom_mask = TensorProperty(shape=(-1,), dtype="bool", device="cuda")
```

---

## Scene Access

The `lichtfeld` module (`lf`) provides access to the scene graph, node operations, selection, and transforms.

### Getting the scene

```python
import lichtfeld as lf

scene = lf.get_scene()          # Get scene object (None if no scene loaded)
if lf.has_scene():
    print(f"Total gaussians: {scene.total_gaussian_count}")
```

### Node operations

```python
scene = lf.get_scene()

# Add nodes
group_id = scene.add_group("My Group")
splat_id = scene.add_splat(
    "My Splat",
    means=lf.Tensor.zeros([100, 3], device="cuda"),
    sh0=lf.Tensor.zeros([100, 1, 3], device="cuda"),
    shN=lf.Tensor.zeros([100, 0, 3], device="cuda"),
    scaling=lf.Tensor.zeros([100, 3], device="cuda"),
    rotation=lf.Tensor.zeros([100, 4], device="cuda"),
    opacity=lf.Tensor.zeros([100, 1], device="cuda"),
)

# Query nodes
nodes = scene.get_nodes()
node = scene.get_node("My Splat")
visible = scene.get_visible_nodes()

# Modify
scene.rename_node("My Splat", "Renamed Splat")
scene.reparent(splat_id, group_id)
scene.remove_node("Renamed Splat", keep_children=False)
new_name = scene.duplicate_node("My Group")
```

### Selection

```python
import lichtfeld as lf

lf.select_node("My Splat")
names = lf.get_selected_node_names()
lf.deselect_all()
has_sel = lf.has_selection()

# Gaussian-level selection (mask-based)
scene = lf.get_scene()
mask = lf.Tensor.zeros([scene.total_gaussian_count], dtype="bool", device="cuda")
mask[0:100] = True
scene.set_selection_mask(mask)
scene.clear_selection()
```

### Transforms

```python
import lichtfeld as lf

# Get/set as 16-float column-major matrix
matrix = lf.get_node_transform("My Splat")
lf.set_node_transform("My Splat", matrix)

# Decompose/compose
components = lf.decompose_transform(matrix)
# components = {"translation": [x,y,z], "euler": [rx,ry,rz], "scale": [sx,sy,sz]}

new_matrix = lf.compose_transform(
    translation=[1.0, 2.0, 3.0],
    euler_deg=[0.0, 45.0, 0.0],
    scale=[1.0, 1.0, 1.0],
)
```

### Splat data access

Splat data can be accessed from the combined model or from individual scene nodes:

```python
scene = lf.get_scene()

# Combined model (all splat nodes merged)
model = scene.combined_model()

# Per-node access
for node in scene.get_nodes():
    sd = node.splat_data()       # None for non-splat nodes
    if sd is not None:
        print(f"{node.name}: {sd.num_points} gaussians")
```

```python
# Raw data (views into GPU memory — no copy)
means = model.means_raw           # [N, 3] positions
sh0 = model.sh0_raw               # [N, 1, 3] base SH coefficients
shN = model.shN_raw               # [N, K, 3] higher-order SH
scaling = model.scaling_raw        # [N, 3] log-space scaling
rotation = model.rotation_raw      # [N, 4] quaternion rotation
opacity = model.opacity_raw        # [N, 1] logit-space opacity

# Activated data (transformed to usable form)
activated_opacity = model.get_opacity()     # sigmoid applied, [N]
activated_scaling = model.get_scaling()     # exp applied
activated_rotation = model.get_rotation()   # normalized quaternions

# Metadata
count = model.num_points
sh_deg = model.active_sh_degree
```

### Soft delete

Soft delete hides gaussians without removing them. After modifying the deletion mask, call `scene.notify_changed()` to update the viewport:

```python
scene = lf.get_scene()
for node in scene.get_nodes():
    sd = node.splat_data()
    if sd is None:
        continue

    # Hide gaussians with opacity below threshold
    opacity = sd.get_opacity()            # [N] in [0, 1]
    mask = opacity < 0.1
    sd.soft_delete(mask)

# Trigger viewport redraw — required after modifying scene data
scene.notify_changed()

# Restore all hidden gaussians
for node in scene.get_nodes():
    sd = node.splat_data()
    if sd is not None:
        sd.clear_deleted()
scene.notify_changed()
```

> **Note:** `scene.invalidate_cache()` only clears the internal cache. It does **not** trigger a viewport redraw. Use `scene.notify_changed()` instead — it invalidates the cache and signals the renderer.

### Example: scene manipulation plugin

```python
import lichtfeld as lf
from lfs_plugins.types import Operator

class SceneInfo(Operator):
    label = "Print Scene Info"

    def execute(self, context) -> set:
        scene = lf.get_scene()
        if scene is None:
            lf.log.warn("No scene loaded")
            return {"CANCELLED"}

        for node in scene.get_nodes():
            bounds = scene.get_node_bounds(node.id)
            lf.log.info(f"Node: {node.name}, bounds: {bounds}")

        return {"FINISHED"}

class CenterSelection(Operator):
    label = "Center Selection"

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_selection() and lf.can_transform_selection()

    def execute(self, context) -> set:
        center = lf.get_selection_world_center()
        if center:
            lf.log.info(f"Selection center: {center}")
        return {"FINISHED"}
```

---

## Signals

Signals provide reactive state management. When a signal's value changes, all subscribers are notified.

### Signal types

```python
from lfs_plugins.ui.signals import Signal, ComputedSignal, ThrottledSignal, Batch

# Basic signal
count = Signal(0, name="count")
count.value = 5                          # Notifies subscribers
current = count.value                    # Read current value
current = count.peek()                   # Read without tracking

# Subscribe
unsub = count.subscribe(lambda v: print(f"Count: {v}"))
unsub()                                  # Stop receiving updates

# Owner-tracked subscription (auto-cleanup on plugin unload)
unsub = count.subscribe_as("my_plugin", lambda v: print(v))

# Computed signal (derived from others)
a = Signal(2)
b = Signal(3)
product = ComputedSignal(lambda: a.value * b.value, [a, b])
print(product.value)                     # 6

# Throttled signal (rate-limited notifications)
iteration = ThrottledSignal(0, max_rate_hz=30)
iteration.value = 1000                   # Only notifies ~30 times/sec
iteration.flush()                        # Force pending notification
```

### Batch context manager

Defer notifications until all updates are complete:

```python
from lfs_plugins.ui.signals import Batch

with Batch():
    state.x.value = 10
    state.y.value = 20
    state.z.value = 30
# Subscribers notified once here, not three times
```

### AppState

Pre-defined signals for application state:

```python
from lfs_plugins.ui.state import AppState

# Training
AppState.is_training              # Signal[bool]
AppState.trainer_state            # Signal[str] - "idle", "ready", "running", "paused", "stopping"
AppState.has_trainer              # Signal[bool]
AppState.iteration                # Signal[int]
AppState.max_iterations           # Signal[int]
AppState.loss                     # Signal[float]
AppState.psnr                     # Signal[float]
AppState.num_gaussians            # Signal[int]

# Scene
AppState.has_scene                # Signal[bool]
AppState.scene_generation         # Signal[int] - increments on scene change
AppState.scene_path               # Signal[str]

# Selection
AppState.has_selection            # Signal[bool]
AppState.selection_count          # Signal[int]
AppState.selection_generation     # Signal[int]

# Viewport
AppState.viewport_width           # Signal[int]
AppState.viewport_height          # Signal[int]

# Computed
AppState.training_progress        # ComputedSignal[float] - 0.0 to 1.0
AppState.can_start_training       # ComputedSignal[bool]
```

### Example: reactive training monitor

```python
import lichtfeld as lf
from lfs_plugins.ui.state import AppState
from lfs_plugins.ui.signals import Signal

class TrainingMonitor(lf.ui.Panel):
    label = "Training Monitor"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 50

    def __init__(self):
        self.best_loss = Signal(float("inf"), name="best_loss")
        self.loss_history = []

        AppState.loss.subscribe_as("my_plugin", self._on_loss_change)

    def _on_loss_change(self, loss: float):
        if loss > 0:
            self.loss_history.append(loss)
            if loss < self.best_loss.value:
                self.best_loss.value = loss

    @classmethod
    def poll(cls, context) -> bool:
        return AppState.has_trainer.value

    def draw(self, ui):
        ui.heading("Training Monitor")

        state = AppState.trainer_state.value
        ui.label(f"State: {state}")
        ui.label(f"Iteration: {AppState.iteration.value}")
        ui.label(f"Loss: {AppState.loss.value:.6f}")
        ui.label(f"Best Loss: {self.best_loss.value:.6f}")
        ui.label(f"PSNR: {AppState.psnr.value:.2f}")
        ui.label(f"Gaussians: {AppState.num_gaussians.value:,}")

        progress = AppState.training_progress.value
        ui.progress_bar(progress, f"{progress * 100:.1f}%")

        if self.loss_history:
            ui.plot_lines(
                "Loss##monitor",
                self.loss_history[-200:],
                0.0, max(self.loss_history[-200:]),
                (0, 80),
            )
```

---

## Capabilities

Capabilities allow plugins to expose features that other plugins (or the application) can invoke.

### CapabilityRegistry

```python
from lfs_plugins.capabilities import CapabilityRegistry, CapabilitySchema
from lfs_plugins.context import PluginContext

registry = CapabilityRegistry.instance()

# Register a capability
def my_handler(args: dict, ctx: PluginContext) -> dict:
    threshold = args.get("threshold", 0.5)
    if ctx.scene:
        # Do something with the scene
        pass
    return {"success": True, "count": 42}

registry.register(
    name="my_plugin.analyze",
    handler=my_handler,
    description="Analyze gaussians by threshold",
    schema=CapabilitySchema(
        properties={"threshold": {"type": "number", "default": 0.5}},
        required=[],
    ),
    plugin_name="my_plugin",
    requires_gui=True,
)

# Invoke a capability
result = registry.invoke("my_plugin.analyze", {"threshold": 0.3})
# result = {"success": True, "count": 42}

# Query
registry.has("my_plugin.analyze")    # True
caps = registry.list_all()           # List[Capability]

# Unregister
registry.unregister("my_plugin.analyze")
registry.unregister_all_for_plugin("my_plugin")
```

### PluginContext

Capability handlers receive a `PluginContext` with scene and view data:

```python
from lfs_plugins.context import PluginContext, SceneContext, ViewContext

def handler(args: dict, ctx: PluginContext) -> dict:
    # Scene access
    if ctx.scene:
        ctx.scene.scene               # PyScene object
        ctx.scene.set_selection_mask(mask)

    # Viewport access
    if ctx.view:
        ctx.view.image                 # [H, W, 3] tensor
        ctx.view.screen_positions      # [N, 2] tensor or None
        ctx.view.width, ctx.view.height
        ctx.view.fov
        ctx.view.rotation              # [3, 3] tensor
        ctx.view.translation           # [3] tensor

    # Invoke other capabilities
    if ctx.capabilities.has("other_plugin.feature"):
        result = ctx.capabilities.invoke("other_plugin.feature", {"key": "value"})

    return {"success": True}
```

---

## Training Hooks

Register callbacks for training lifecycle events.

### Decorators

```python
import lichtfeld as lf

@lf.on_training_start
def on_start():
    lf.log.info("Training started")

@lf.on_iteration_start
def on_iter():
    pass

@lf.on_pre_optimizer_step
def on_pre_opt():
    pass

@lf.on_post_step
def on_post():
    ctx = lf.context()
    if ctx.iteration % 1000 == 0:
        lf.log.info(f"Iteration {ctx.iteration}, loss: {ctx.loss:.6f}")

@lf.on_training_end
def on_end():
    lf.log.info(f"Training finished: {lf.finish_reason()}")
```

### Training context

```python
ctx = lf.context()
ctx.iteration          # Current iteration (int)
ctx.max_iterations     # Target iterations (int)
ctx.loss               # Current loss (float)
ctx.num_gaussians      # Gaussian count (int)
ctx.is_refining        # Currently refining (bool)
ctx.is_training        # Training active (bool)
ctx.is_paused          # Training paused (bool)
ctx.phase              # Current phase (str)
ctx.strategy           # Training strategy (str)
ctx.refresh()          # Update snapshot
```

### Training control

```python
import lichtfeld as lf

lf.start_training()
lf.pause_training()
lf.resume_training()
lf.stop_training()
lf.reset_training()
lf.save_checkpoint()
```

### Example: custom training callback

```python
import lichtfeld as lf
from lfs_plugins.ui.state import AppState

class AutoSavePlugin:
    """Automatically save checkpoints every N iterations."""

    def __init__(self, interval=5000):
        self.interval = interval
        self.last_save = 0

    def on_post_step(self):
        ctx = lf.context()
        if ctx.iteration - self.last_save >= self.interval:
            lf.save_checkpoint()
            self.last_save = ctx.iteration
            lf.log.info(f"Auto-saved at iteration {ctx.iteration}")

_auto_save = None

def on_load():
    global _auto_save
    _auto_save = AutoSavePlugin(interval=5000)
    lf.on_post_step(_auto_save.on_post_step)
    lf.log.info("Auto-save plugin loaded")

def on_unload():
    global _auto_save
    _auto_save = None
```

---

## Hot Reload & Debugging

### File watcher

When `hot_reload = true` in `pyproject.toml`, LichtFeld watches your plugin directory for changes. On any `.py` file save, the plugin is automatically unloaded and reloaded.

### Logging

```python
import lichtfeld as lf

lf.log.info("Informational message")
lf.log.warn("Warning message")
lf.log.error("Error message")
lf.log.debug("Debug message")    # Only visible with --log-level debug
```

### Plugin state inspection

```python
from lfs_plugins.manager import PluginManager

mgr = PluginManager.instance()
state = mgr.get_state("my_plugin")       # PluginState enum
error = mgr.get_error("my_plugin")       # Error message or None
tb = mgr.get_traceback("my_plugin")      # Traceback string or None
```

Or via the `lf` module:

```python
import lichtfeld as lf

lf.plugins.get_state("my_plugin")
lf.plugins.get_error("my_plugin")
lf.plugins.get_traceback("my_plugin")
```

---

## IDE Setup

### Auto-generated pyrightconfig.json

LichtFeld generates a `pyrightconfig.json` in the project root that includes the correct Python paths for type checking.

### VS Code

Add to `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "/path/to/gaussian-splatting-cuda/src/python",
        "/path/to/gaussian-splatting-cuda/build/src/python/typings"
    ]
}
```

### Type stubs

Type stubs are generated at `build/src/python/typings/` and provide autocomplete for:
- `lichtfeld` - Main API (scene, training, rendering, etc.)
- `lichtfeld.ui` - UI functions
- `lichtfeld.scene` - Scene types
- `lichtfeld.selection` - Selection types
- `lichtfeld.plugins` - Plugin management

The committed SDK stubs live in `src/python/stubs/` and are checked against the generated output during the build. If you intentionally change the Python API surface, refresh the committed stubs with:

```bash
cmake --build build --target refresh_python_stubs
```

You can also run the check explicitly with:

```bash
cmake --build build --target check_python_stubs
```

### debugpy attach

Add to your plugin's `on_load()` for VS Code debugging:

```python
def on_load():
    try:
        import debugpy
        debugpy.listen(5678)
        lf.log.info("debugpy listening on port 5678")
    except ImportError:
        pass
```

VS Code launch config:

```json
{
    "name": "Attach to LichtFeld Plugin",
    "type": "debugpy",
    "request": "attach",
    "connect": {"host": "localhost", "port": 5678}
}
```

---

## Installing & Publishing

### Create a new plugin

```python
import lichtfeld as lf

path = lf.plugins.create("my_new_plugin")
```

That Python API creates the minimal source package only. If you also want a plugin venv and editor config, use:

```bash
LichtFeld-Studio plugin create my_new_plugin
```

Both scaffold paths start with the same step-1 panel template and now include `main_panel.rml` and `main_panel.rcss` up front. You can ignore those files until you move into the custom-template styling path.

### Install from GitHub

```python
import lichtfeld as lf

lf.plugins.install("owner/repo")
lf.plugins.install("https://github.com/owner/repo")
```

### Plugin registry

```python
import lichtfeld as lf

results = lf.plugins.search("neural rendering")
lf.plugins.install_from_registry("plugin_id")
lf.plugins.check_updates()
lf.plugins.update("my_plugin")
```

Registry installs use the same v1 compatibility contract as local plugins. A version is eligible only if its `plugin_api`, `lichtfeld_version`, and `required_features` match the current host.

### Manage plugins

```python
import lichtfeld as lf

lf.plugins.discover()              # Scan for installed plugins
lf.plugins.load("my_plugin")       # Load a specific plugin
lf.plugins.unload("my_plugin")     # Unload
lf.plugins.reload("my_plugin")     # Reload (hot reload)
lf.plugins.uninstall("my_plugin")  # Remove
lf.plugins.list_loaded()           # Show loaded plugins
```

### pyproject.toml packaging requirements

For publishing, ensure your `pyproject.toml` includes:
- `name` - Unique plugin identifier
- `version` - Semantic version (e.g., `"1.0.0"`)
- `description` - Clear description of what the plugin does
- `authors` - Your name or organization
- `dependencies` - Any Python dependencies
- `plugin_api` - Supported plugin API range, such as `"~=1.0"` or `">=1,<2"`
- `lichtfeld_version` - Supported host/runtime range, such as `">=0.4.2"`
- `required_features` - Optional host features the plugin requires
