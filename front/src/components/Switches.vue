<template>
  <div class="switch">
    <label class="switch-label">
      <input
        :checked="modelValue"
        type="checkbox"
        class="switch-input"
        :value="modelValue"
        @input="updateModelValue"
      />
      <span class="switch-slider"></span>
    </label>
  </div>
</template>

<script setup>
import { defineProps, defineEmits } from 'vue'

const emit = defineEmits(['update:modelValue'])
const props = defineProps({
  modelValue: {
    type: Boolean,
    default: false
  },

  index: {
    type: Number
  }
})

function updateModelValue() {
  emit('statusSwitch', props.index)
}
</script>

<style scoped>
.switch {
  cursor: pointer;
}

.switch-input {
  display: none;
}

.switch-slider {
  position: relative;
  display: inline-block;
  width: 40px;
  height: 20px;
  background-color: var(--background-color);
  border-radius: 20px;
  margin-right: 8px;
  transition: background-color 0.2s ease-in-out;
}

.switch-slider:before {
  content: '';
  position: absolute;
  width: 16px;
  height: 16px;
  background-color: var(--text-color-white);
  border-radius: 50%;
  left: 2px;
  top: 2px;
  transition: transform 0.2s ease-in-out;
}

.switch-input:checked + .switch-slider {
  background-color: var(--main-color);
  transition: background-color 0.2s ease-in-out;
}

.switch-input:checked + .switch-slider:before {
  transform: translateX(20px);
}
</style>
