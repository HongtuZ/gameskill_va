"""Shared action schema constants."""

DEFAULT_KEYBOARD_ACTIONS: tuple[str, ...] = (
    "Fire",
    "AltFire",
    "Ping",
    "CyclePrimaryWeaponNext",
    "CyclePrimaryWeaponPrev",
    "MoveForward",
    "MoveBackward",
    "MoveLeft",
    "MoveRight",
    "Jump",
    "Crouch",
    "Walk",
    "Activate_Ability1",
    "Activate_Ability2",
    "Activate_Ability3",
    "Activate_Ultimate",
    "UseObject",
    "Reload",
    "EquipPrimaryWeapon",
    "EquipSecondaryWeapon",
    "EquipMeleeWeapon",
    "EquipSpike",
    "DropEquippable",
    "InspectWeapon",
    "OpenWheel",
    "RadioCommsMenu",
    "PushToTalk",
    "OpenMap",
    "ToggleScoreboard",
    "OpenShop",
    "Esc",
)

MOUSE_ACTION_DIM = 2
HYBRID_ACTION_DIM = MOUSE_ACTION_DIM + len(DEFAULT_KEYBOARD_ACTIONS)

__all__ = [
    "DEFAULT_KEYBOARD_ACTIONS",
    "HYBRID_ACTION_DIM",
    "MOUSE_ACTION_DIM",
]
