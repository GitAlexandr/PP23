<template>
  <button @click="toggleRecording">
    <SvgIcon class="mdi" type="mdi" :path="mdiMicrophone" />
  </button>
</template>

<script>
import { ref, onUnmounted } from 'vue';
import { useStore } from 'vuex';

import SvgIcon from '@jamescoyle/vue-icon';
import { mdiMicrophone } from '@mdi/js';

export default {
  components: {
    SvgIcon,
  },
  setup() {
    const store = useStore();
    const recording = ref(false);
    const audioChunks = ref([]);
    let mediaRecorder;
    let audioStream;

    const startRecording = async () => {
      try {
        audioChunks.value = [];
        audioStream = await navigator.mediaDevices.getUserMedia({audio: true});
        mediaRecorder = new MediaRecorder(audioStream);

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            audioChunks.value.push(e.data);
          }
        };

        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunks.value, {type: 'audio/wav'});
          const audioUrl = URL.createObjectURL(audioBlob);

          store.dispatch('sendAMessageVoice', audioUrl)
          // store.dispatch('chatHistory');
        }

        mediaRecorder.start();
        recording.value = true;
      } catch (error) {
        console.error('Error accessing microphone:', error);
      }
    };

    const stopRecording = () => {
      if (mediaRecorder && recording.value) {
        mediaRecorder.stop();
        audioStream.getTracks().forEach((track) => track.stop());
        recording.value = false;
      }
    };

    const toggleRecording = () => {
      if (recording.value) {
        stopRecording();
      } else {
        startRecording();
      }
    };

    onUnmounted(() => {
      stopRecording();
    });

    return {
      recording,
      toggleRecording,
      mdiMicrophone,
    };
  },
};
</script>

<style scoped lang="scss">
button {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: 4px;
  width: 24px;
  height: 24px;
  background: var(--text-color-dark-grey);
  border: 0;
  border-radius: 50%;
  color: var(--background-color);
}

.mdi {
  width: 20px !important;
  height: 20px !important;
  font-size: 20px;
}
</style>


