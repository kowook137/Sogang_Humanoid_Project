import threading


class KeyboardCommandController:
    """Persistent SE(2) velocity commands controlled from the MuJoCo viewer."""

    def __init__(
        self,
        linear_step: float = 0.1,
        yaw_step: float = 0.2,
        max_linear_velocity: float = 0.5,
        max_yaw_velocity: float = 1.0,
    ) -> None:
        self.linear_step = linear_step
        self.yaw_step = yaw_step
        self.max_linear_velocity = max_linear_velocity
        self.max_yaw_velocity = max_yaw_velocity

        self._velocity = [0.0, 0.0, 0.0]
        self._lock = threading.Lock()

        self.print_help()

    @staticmethod
    def _clip(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def print_help(self) -> None:
        print(
            "\nKeyboard controls:\n"
            "  W/S : forward/backward\n"
            "  A/D : left/right\n"
            "  Q/E : turn left/right\n"
            "  X or Space : stop\n"
            "  H : show this help\n"
        )

    def handle_key(self, keycode: int) -> None:
        try:
            key = chr(keycode).lower()
        except (TypeError, ValueError):
            return

        if key == "h":
            self.print_help()
            return

        with self._lock:
            if key == "w":
                self._velocity[0] += self.linear_step
            elif key == "s":
                self._velocity[0] -= self.linear_step
            elif key == "a":
                self._velocity[1] += self.linear_step
            elif key == "d":
                self._velocity[1] -= self.linear_step
            elif key == "q":
                self._velocity[2] += self.yaw_step
            elif key == "e":
                self._velocity[2] -= self.yaw_step
            elif key in {"x", " "}:
                self._velocity = [0.0, 0.0, 0.0]
            else:
                return

            self._velocity[0] = self._clip(
                self._velocity[0], self.max_linear_velocity
            )
            self._velocity[1] = self._clip(
                self._velocity[1], self.max_linear_velocity
            )
            self._velocity[2] = self._clip(
                self._velocity[2], self.max_yaw_velocity
            )
            command = tuple(self._velocity)

        print(
            f"Keyboard command: "
            f"vx={command[0]:+.2f}, "
            f"vy={command[1]:+.2f}, "
            f"yaw={command[2]:+.2f}"
        )

    def get_velocity(self) -> tuple[float, float, float]:
        with self._lock:
            return tuple(self._velocity)
