LICENSE = "CLOSED"
SUMMARY = "Simple test application"
SECTION = "onnxruntime/apps"
SRCREV = "${AUTOREV}"
S = "${WORKDIR}/git"

SRC_URI = " \
            git://gh.llkk.cc/https://github.com/NaOHCC/onnx-app.git;branch=master;protocol=https \
          "


inherit cmake

DEPENDS += "onnxruntime opencv"

RDEPENDS:${PN} += " \
    onnxruntime \
    opencv \
"

# do_install:append() {
#     mkdir -p ${D}${datadir}/model/
#     install -m 644 ${WORKDIR}/model.torch.qat.onnx ${D}${datadir}/model/
# }

FILES:${PN} += " \
    ${bindir}/onnx-app \
"