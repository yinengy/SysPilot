import pytest
from syspilot.skills.base import SkillRegistry, SkillResult
from syspilot.skills.builtin import DynamicScriptSkill


def test_skill_registry_singleton():
    registry1 = SkillRegistry()
    registry2 = SkillRegistry()
    assert registry1 is registry2


def test_skill_registration():
    registry = SkillRegistry()
    skill = DynamicScriptSkill()
    registry.register(skill)
    
    retrieved = registry.get("dynamic_script")
    assert retrieved is skill


@pytest.mark.asyncio
async def test_dynamic_script_python():
    skill = DynamicScriptSkill()
    
    result = await skill.execute(
        script_type="python",
        script="print('hello world')",
    )
    
    assert result.success
    assert "hello world" in result.output


@pytest.mark.asyncio
async def test_dynamic_script_bash():
    skill = DynamicScriptSkill()
    
    result = await skill.execute(
        script_type="bash",
        script="echo 'test output'",
    )
    
    assert result.success
    assert "test output" in result.output


@pytest.mark.asyncio
async def test_dynamic_script_empty():
    skill = DynamicScriptSkill()
    
    result = await skill.execute(
        script_type="python",
        script="",
    )
    
    assert not result.success
    assert "Empty script" in result.error
