<template>
  <div>
    <button class="button button-skills" @click="openListSkills">Навыки</button>
    <div v-if="dialogs" class="dialogs" @click.self="closeListSkills">
      <div class="list-skills">
        <div class="skill" v-for="(button, index) in buttons" :key="index">
          <div class="name-skill">{{ button.title }}</div>
          <switches :modelValue="button.status" :index="index" @statusSwitch="handleStatusSwitch(index)" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from "vue"
import { useStore } from "vuex"
import Switches from "@/components/Switches.vue"

const dialogs = ref(false)
const store = useStore()
const buttons = computed(() => store.getters.getManagementButtons)

function openListSkills() {
  dialogs.value = !dialogs.value
}

function closeListSkills() {
  dialogs.value = false
}

function handleStatusSwitch(index) {
  store.commit('statusSwitch', index)
}
</script>

<style scoped lang="scss">
.button {
  margin: 4px 0;
  padding: 12px;
  background: var(--main-color);
  color: var(--text-color-white);
  border: 0;
  border-radius: 16px;
  font-size: 16px;
  box-shadow: 0 4px 6px var(--box-shadow-color);

  &:hover {
    transform: scale(1.01);
  }

  &:active {
    transform: scale(0.99);
  }

  &-skills {
    width: 100%;
  }
}

.dialogs {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 2;
  backdrop-filter: blur(5px);
}

.list-skills {
  background: var(--text-color-white);
  padding: 16px;
  border-radius: 16px;
  width: 500px;

  .skill {
    border-radius: 8px;
    padding: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;

    &:hover {
      background: var(--background-hover);
    }

    .name-skill {
      margin-left: 4px;
    }
  }
}
</style>
