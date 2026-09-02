import { common } from '@kit.AbilityKit';
import { image } from '@kit.ImageKit';
import { photoAccessHelper } from '@kit.MediaLibraryKit';
import native from 'libentry.so';

class GearSpec {
  name: string;
  width: number;
  height: number;

  constructor(name: string, width: number, height: number) {
    this.name = name;
    this.width = width;
    this.height = height;
  }
}

const GEARS: GearSpec[] = [
  new GearSpec('384x512', 384, 512),
  new GearSpec('576x768', 576, 768),
  new GearSpec('768x1024', 768, 1024),
  new GearSpec('960x1408', 960, 1408)
];

function selectGear(width: number, height: number): GearSpec {
  const sourceArea: number = width * height;
  for (let index: number = 0; index < GEARS.length; index++) {
    const gear: GearSpec = GEARS[index];
    if (sourceArea <= gear.width * gear.height) {
      return gear;
    }
  }
  return GEARS[GEARS.length - 1];
}

@Entry
@Component
struct Index {
  @State status: string = '请选择图片，或直接运行仓库内置验证图';
  @State resultText: string = '';
  @State selectedLabel: string = '内置 images/verify.png';
  @State running: boolean = false;

  private async runSource(source: image.ImageSource): Promise<void> {
    this.running = true;
    this.resultText = '';
    let pixelMap: image.PixelMap | undefined = undefined;
    try {
      pixelMap = await source.createPixelMap({
        desiredPixelFormat: image.PixelMapFormat.BGRA_8888
      });
      const originalInfo: image.ImageInfo = await pixelMap.getImageInfo();
      const gear: GearSpec = selectGear(originalInfo.size.width, originalInfo.size.height);
      const scale: number = Math.min(
        gear.width / originalInfo.size.width,
        gear.height / originalInfo.size.height
      );
      const scaledWidth: number = Math.max(1, Math.min(gear.width,
        Math.round(originalInfo.size.width * scale)));
      const scaledHeight: number = Math.max(1, Math.min(gear.height,
        Math.round(originalInfo.size.height * scale)));
      this.status = `预处理 ${originalInfo.size.width}×${originalInfo.size.height} → ` +
        `${scaledWidth}×${scaledHeight}，动态档位 ${gear.name}`;
      await pixelMap.scale(
        scaledWidth / originalInfo.size.width,
        scaledHeight / originalInfo.size.height
      );

      const buffer: ArrayBuffer = new ArrayBuffer(scaledWidth * scaledHeight * 4);
      await pixelMap.readPixelsToBuffer(buffer);
      const context: common.UIAbilityContext =
        this.getUIContext().getHostContext() as common.UIAbilityContext;
      this.status = `NPU 推理中（${gear.name}，最多 256 token）`;
      const result = await native.runInference(
        context.resourceManager,
        new Uint8Array(buffer),
        scaledWidth,
        scaledHeight,
        gear.name,
        256
      );
      if (!result.ok) {
        this.status = `失败 ${result.errorCode}: ${result.message}`;
        return;
      }
      this.resultText = result.text;
      this.status = `完成，动态档位 ${result.gear}`;
    } catch (error) {
      const failure: Error = error as Error;
      this.status = `ArkTS 层失败: ${failure.message}`;
    } finally {
      if (pixelMap !== undefined) {
        pixelMap.release();
      }
      source.release();
      this.running = false;
    }
  }

  private async runBundledImage(): Promise<void> {
    const context: common.UIAbilityContext =
      this.getUIContext().getHostContext() as common.UIAbilityContext;
    const content: Uint8Array = await context.resourceManager.getRawFileContent('images/verify.png');
    const copied: Uint8Array = new Uint8Array(content.length);
    copied.set(content);
    this.selectedLabel = '内置 images/verify.png';
    await this.runSource(image.createImageSource(copied.buffer));
  }

  private async pickAndRun(): Promise<void> {
    try {
      const picker: photoAccessHelper.PhotoViewPicker = new photoAccessHelper.PhotoViewPicker();
      const options: photoAccessHelper.PhotoSelectOptions = new photoAccessHelper.PhotoSelectOptions();
      options.MIMEType = photoAccessHelper.PhotoViewMIMETypes.IMAGE_TYPE;
      options.maxSelectNumber = 1;
      const selection: photoAccessHelper.PhotoSelectResult = await picker.select(options);
      if (selection.photoUris.length === 0) {
        this.status = '未选择图片';
        return;
      }
      this.selectedLabel = selection.photoUris[0];
      await this.runSource(image.createImageSource(selection.photoUris[0]));
    } catch (error) {
      const failure: Error = error as Error;
      this.status = `选图失败: ${failure.message}`;
    }
  }

  build() {
    Scroll() {
      Column({ space: 16 }) {
        Text('UniRec-0.1B · Dynamic OM')
          .fontSize(26)
          .fontWeight(FontWeight.Bold)
          .width('100%');

        Text('单 encoder.om + 单 decoder.om，四档 NPU 推理，禁止 CPU 回退')
          .fontSize(14)
          .fontColor('#5F6368')
          .width('100%');

        Text(this.selectedLabel)
          .fontSize(12)
          .fontColor('#6A6A6A')
          .maxLines(2)
          .width('100%');

        Row({ space: 12 }) {
          Button('运行内置验证图')
            .enabled(!this.running)
            .layoutWeight(1)
            .onClick(() => { this.runBundledImage(); });
          Button('选择图片')
            .enabled(!this.running)
            .layoutWeight(1)
            .onClick(() => { this.pickAndRun(); });
        }
        .width('100%');

        Column({ space: 8 }) {
          Text('状态')
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .width('100%');
          Text(this.status)
            .fontSize(14)
            .fontColor(this.status.startsWith('失败') ? '#B3261E' : '#1F1F1F')
            .width('100%');
        }
        .padding(16)
        .borderRadius(12)
        .backgroundColor('#F2F4F7')
        .width('100%');

        Column({ space: 8 }) {
          Text('识别结果')
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .width('100%');
          Text(this.resultText.length > 0 ? this.resultText : '尚无结果')
            .fontSize(16)
            .copyOption(CopyOptions.LocalDevice)
            .width('100%');
        }
        .padding(16)
        .borderRadius(12)
        .backgroundColor('#FFFFFF')
        .border({ width: 1, color: '#E3E5E8' })
        .width('100%');
      }
      .padding(20)
      .width('100%');
    }
    .backgroundColor('#FAFAFA')
    .width('100%')
    .height('100%');
  }
}
