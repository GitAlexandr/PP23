<template>
  <div class="container">
    <div class="ava" ref="avaRef">
      <transition name="avatar-transition">
        <img :src="avatar" alt="Аватар" key="avatar" />
      </transition>
    </div>
    <div class="buttons">
      <div>
        <input class="hide" type="file" ref="fileInput" @change="handleFileChange" />
        <button class="button file-input" @click="uploadFile">Загрузить файл</button>
      </div>
      <button class="button" @click="clearMessages">Очистить</button>
      <ListOfSkills />
    </div>
  </div>
</template>

<script setup>
import { useStore } from "vuex";
import { ref, onMounted } from 'vue';
import Hammer from 'hammerjs';
import ListOfSkills from '@/components/List-of-skills.vue';

const store = useStore();
const fileInput = ref(null);
const avatarIndex = ref(0);
const avatar = ref('');
const avatarElement = ref(null);

async function loadImageArray() {
  const context = import.meta.glob('@/assets/ava/*');
  const keys = Object.keys(context);
  const files = await Promise.all(keys.map(key => context[key]()));

  return files.filter(file => file.default);
}

function clearMessages() {
  store.commit('clearMessage');
}

function handleFileChange() {
  store.dispatch('sendAMessageFile', fileInput.value.files[0]);
}

function uploadFile() {
  fileInput.value.click();
}

onMounted(async () => {
  const avaImagesArray = await loadImageArray();
  avatarElement.value = document.querySelector('.ava');

  if (avatarElement.value) {
    const avatarSwipeHandler = new Hammer.Manager(avatarElement.value);
    avatarSwipeHandler.add(new Hammer.Swipe({direction: Hammer.DIRECTION_HORIZONTAL}));

    avatarSwipeHandler.on('swiperight swipeleft', (e) => {
      if (e.type === 'swiperight' && avatarIndex.value < avaImagesArray.length - 1) {
        // Свайп вправо
        avatarIndex.value += 1;
      } else if (e.type === 'swipeleft' && avatarIndex.value > 0) {
        // Свайп влево
        avatarIndex.value -= 1;
      }

      avatar.value = avaImagesArray[avatarIndex.value].default;
    });

    avatarSwipeHandler.get('swipe').set({enable: true});
  }
});
</script>

<style scoped lang="scss">
.container {
  width: 220px;
  padding: 16px;
  border-radius: 16px;
  background: var(--background-color);
  display: flex;
  flex-direction: column;
  box-shadow: 0 4px 6px var(--box-shadow-color);

  .ava {
    display: flex;
    justify-content: center;

    img {
      width: 200px;
      height: 200px;
      border-radius: 50%;
      border: 4px solid var(--main-color);
      box-shadow: 0 4px 6px var(--box-shadow-color);
      pointer-events: none;
    }
  }

  .buttons {
    margin-top: 20px;
    display: flex;
    flex-direction: column;

    .file-input {
      width: 100%;
    }

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
    }
  }
}
</style>
