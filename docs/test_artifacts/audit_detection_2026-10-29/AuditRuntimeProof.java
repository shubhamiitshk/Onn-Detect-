import com.onndetect.detection.provenance.*;
import com.onndetect.detection.video.VideoScorePolicy;
import com.onndetect.inference.ModelIntegrityChecker;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import kotlin.Pair;
import org.bouncycastle.asn1.x500.X500Name;
import org.bouncycastle.cert.jcajce.JcaX509CertificateConverter;
import org.bouncycastle.cert.jcajce.JcaX509v3CertificateBuilder;
import org.bouncycastle.operator.jcajce.JcaContentSignerBuilder;

/** Audit-only proof invoking the already-built application classes. No production edits. */
public class AuditRuntimeProof {
    static final C2paTestFixtures F = C2paTestFixtures.INSTANCE;

    static C2paTestFixtures.JpegLayout layout(byte[] unsignedClaim, byte[] unchangedCose,
            C2paTestFixtures.TestCa ca, boolean mutateCoveredByte) throws Exception {
        C2paTestFixtures.JpegLayout prev = null;
        for (int pass = 0; pass < 12; pass++) {
            List<Pair<Long, Long>> exclusions = List.of(new Pair<>(
                prev == null ? 0L : prev.getManifestStart(),
                prev == null ? 4096L : prev.getManifestEnd()));
            byte[] digest = F.sha256(prev == null ? new byte[0] : prev.nonManifestBytes());
            String hashJson = F.dataHashJson(exclusions, digest);
            byte[] store = F.buildStore(unsignedClaim, unchangedCose,
                List.of(ca.getCertificate().getEncoded()),
                List.of(new Pair<>("jumd", ("c2pa.hash.data" + hashJson).getBytes(StandardCharsets.UTF_8))),
                "c2pa");
            C2paTestFixtures.JpegLayout next = F.jpegWithApp11(store, 24, 2048,
                new byte[] { (byte) 0xff, (byte) 0xd9 });
            if (mutateCoveredByte) next.getBytes()[40] ^= 0x5a;
            if (prev != null && Arrays.equals(prev.getBytes(), next.getBytes())) return next;
            prev = next;
        }
        throw new AssertionError("Fixture did not converge");
    }

    static ProvenanceEvidence.C2PAClaim verify(C2paTestFixtures.JpegLayout layout,
            C2paTestFixtures.TestCa ca) {
        C2PAVerifier verifier = new C2PAVerifier(new C2PAManifestChecker(), new C2PATrustStore(ca.pem()));
        return (ProvenanceEvidence.C2PAClaim) verifier.verifyStream(
            new ByteArrayInputStream(layout.getBytes())).getEvidence();
    }

    public static void main(String[] args) throws Exception {
        Path repo = Path.of(args[0]);
        C2paTestFixtures.TestCa ca = F.newCa("Ephemeral audit CA");
        byte[] signedCaptureClaim = F.buildClaim("digitalCapture");
        byte[] unchangedCose = F.buildCoseSign1(signedCaptureClaim, ca, null);

        var original = layout(signedCaptureClaim, unchangedCose, ca, false);
        var originalResult = verify(original, ca);
        var changedClaim = layout(F.buildClaim("trainedAlgorithmicMedia"), unchangedCose, ca, false);
        var changedClaimResult = verify(changedClaim, ca);
        var changedMedia = layout(signedCaptureClaim, unchangedCose, ca, true);
        var changedMediaResult = verify(changedMedia, ca);

        System.out.println("PROOF: synthetic ephemeral CA fixtures, not real signed C2PA media conformance");
        System.out.println("signed_payload_source=digitalCapture; identical COSE reused for every case");
        System.out.println("original: signed=" + originalResult.getSigned() + "; source=" + originalResult.getDigitalSourceType());
        System.out.println("unsigned_clai_changed: signed=" + changedClaimResult.getSigned()
            + "; source=" + changedClaimResult.getDigitalSourceType()
            + "; confirmsAI=" + changedClaimResult.getConfirmsAiInvolvement());
        System.out.println("covered_media_byte_changed_and_unsigned_hash_recomputed: signed="
            + changedMediaResult.getSigned() + "; sameCOSE=true; mediaSHAChanged="
            + !Arrays.equals(F.sha256(original.getBytes()), F.sha256(changedMedia.getBytes())));

        var leafGenerator = java.security.KeyPairGenerator.getInstance("EC");
        leafGenerator.initialize(new java.security.spec.ECGenParameterSpec("secp256r1"));
        var leafKey = leafGenerator.generateKeyPair();
        var issuerName = new X500Name(ca.getCertificate().getSubjectX500Principal().getName());
        var expiredBuilder = new JcaX509v3CertificateBuilder(issuerName, java.math.BigInteger.valueOf(42),
            new Date(System.currentTimeMillis() - 172800000L), new Date(System.currentTimeMillis() - 86400000L),
            new X500Name("CN=Expired audit leaf"), leafKey.getPublic());
        var expiredLeaf = new JcaX509CertificateConverter().getCertificate(expiredBuilder.build(
            new JcaContentSignerBuilder("SHA256withECDSA").build(ca.getKeyPair().getPrivate())));
        boolean expired = false;
        try { expiredLeaf.checkValidity(); } catch (java.security.cert.CertificateExpiredException e) { expired = true; }
        boolean chainAccepted = new C2PATrustStore(ca.pem()).verifyChain(expiredLeaf, List.of(ca.getCertificate()));
        System.out.println("expired_leaf: checkValidityExpired=" + expired + "; verifyChainAccepted=" + chainAccepted);

        float score = VideoScorePolicy.INSTANCE.combineMultimodalScore(
            .55f, .57819986f, 1f, false, .35546875f, Float.NaN, 0f, Float.NaN, false);
        System.out.println("unvalidated_audio_boost: visual=.55; audio=1; audioValidatedSuccess=false; result="
            + score + "; crossesVisualThreshold=" + (score >= .57819986f));

        for (String model : List.of(ModelIntegrityChecker.IMAGE_MODEL, ModelIntegrityChecker.AUDIO_MODEL)) {
            var report = ModelIntegrityChecker.INSTANCE.verifyModelsDetailed(Set.of(model), path -> {
                try { return Files.newInputStream(repo.resolve("android/app/src/main/assets").resolve(path)); }
                catch (IOException e) { throw new UncheckedIOException(e); }
            });
            System.out.println("stock_gate " + model + ": allOk=" + report.getAllOk()
                + "; validated=" + report.getValidated() + "; deviceValidated=" + report.getDeviceValidated()
                + "; cascadeValidated=" + report.getCascadeValidated() + "; threshold=" + report.getThreshold()
                + "; failures=" + report.getFailures());
        }
        if (!changedClaimResult.getSigned() || !changedClaimResult.getConfirmsAiInvolvement()
                || !changedMediaResult.getSigned() || score < .57819986f || !expired || !chainAccepted) {
            throw new AssertionError("Expected audit vulnerability reproduction did not occur");
        }
    }
}
