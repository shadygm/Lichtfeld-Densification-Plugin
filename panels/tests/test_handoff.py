"""Worker/UI handoff cannot discard a final result or its training flag."""
from threading import Event, Lock, Thread, current_thread, main_thread

from async_host import AsyncPanelTestCase


class HandoffTests(AsyncPanelTestCase):
    def test_final_result_arriving_during_preview_take_survives(self):
        go, attempting, finished = Event(), Event(), Event()
        errors = []
        gate = Lock()
        case = self

        class HandoffLock:
            def __enter__(self):
                if current_thread() is not main_thread():
                    attempting.set()
                gate.acquire()
                return self

            def __exit__(self, *args):
                gate.release()
                if current_thread() is main_thread() and attempting.is_set():
                    case.assertTrue(finished.wait(2))

        class RacingPanel(self.Panel):
            @property
            def _pending_import(self):
                output = self.__dict__.get('_output')
                if current_thread() is main_thread() and self.__dict__.get('_armed'):
                    self.__dict__['_armed'] = False
                    go.set()
                    case.assertTrue(attempting.wait(2))
                return output

            @_pending_import.setter
            def _pending_import(self, value):
                self.__dict__['_output'] = value

        self.panel.__class__ = RacingPanel
        self.panel._handoff_lock = HandoffLock()
        self.panel._on_cloud_preview(self.result().cloud)
        self.panel._start_training_when_complete = True
        final = self.result()

        def complete():
            try:
                if not go.wait(2):
                    raise RuntimeError('UI did not consume preview')
                self.panel._on_complete(final)
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()

        worker = Thread(target=complete)
        worker.start()
        self.panel._armed = True
        try:
            self.panel.on_update(None)
        finally:
            go.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.panel._cloud_update.kind, 'preview')
        self.assertIs(self.panel._pending_import, final)
        self.assertTrue(self.panel._pending_start_training)
        self.panel.on_update(None)
        self.assertEqual(self.panel._cloud_update.kind, 'final')
        self.assertTrue(self.panel._cloud_update.start_training)
        self.panel._cloud_update.ticket.publish()
        self.panel.on_update(None)
        self.assertIs(self.panel.last_result, final)
        self.host.start_training.assert_called_once()

    def test_native_submission_does_not_hold_the_handoff_lock(self):
        submit = self.scene.node.point_cloud().set_data_async
        def checked_submit(*args, **kwargs):
            self.assertTrue(self.panel._handoff_lock.acquire(blocking=False))
            self.panel._handoff_lock.release()
            return submit(*args, **kwargs)
        self.scene.node.point_cloud().set_data_async = checked_submit
        self.submit_final()
